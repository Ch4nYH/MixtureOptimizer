import torch
from utils import accuracy, AverageMeter
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage

class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, USE_CUDA=True, \
        unroll_length = 5, meta = False, train_loader =None, val_loader = None, print_freq = 5, \
        writer = None, rollout = None, meta_info = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.iterations = 0
        self.USE_CUDA = USE_CUDA
        self.unroll_length = unroll_length
        self.stats = {}
        self.meta = meta
        self.train_loader = train_loader
        self.val_loader = val_dataset
        self.train_acc = AverageMeter()
        self.val_acc   = AverageMeter()
        self.print_preq = print_freq
        self.writer = writer
        self.rollout = rollout
        self.reward_baseline = 0
        self.rewards = []
        self.meta_info = meta_info


        self.window_size = 5
        self.iter_train_loader = iter(self.train_loader)
        
        if USE_CUDA:
            self.model = self.model.cuda()

    def run(self, epochs=1):
        for i in range(1, epochs + 1):
            self.train_acc.reset()
            self.val_acc.reset()
            print("Epoch [{}/{}]: Training ... ".format(i, epochs))
            self.train()
            self.val()
            self.writer.add_scalar('train/accuracy', self.train_acc.avg, i)
            self.writer.add_scalar('test/accuracy', self.val_acc.avg, i)
    
    def reward(self):
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * self.val_acc.avg / 100
        return self.val_acc.avg / 100 - self.reward_baseline
    
    def train(self):
        current_optimizee_step, prev_optimizee_step = 0, 0
        self.model.train()
        for i, data in enumerate(tqdm(self.dataset)):
            batch_input, batch_target = data
            input_var = batch_input
            target_var = batch_target
            if USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            batch_output = self.model(input_var)
            loss = self.criterion(batch_output, target_var)
            acc = accuracy(batch_output, target_var)
            self.train_acc.update(acc)
            if (i % self.print_preq == 0):
                print("Train Accuracy: {:.4f}({:.4f})".format(acc, self.train_acc.avg))
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.meta and i % self.unroll_length == 0:
                self.val_acc.reset()
                self.val()
                reward = self.reward()
                self.optimizer.meta_update(reward)
            
        self.iterations += i

    def train_step(self): 
        losses = []
        fc_mean = []
        fc_std = []
        optimizee_step = []
        for idx in range(self.window_size):
            self.optimizer.zero_grad()
            input, label = self.get_samples()
            if self.USE_CUDA:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            loss = self.criterion(output, label.long())
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())
        losses = [sum(losses) / len(losses)]
        observation = torch.stack(losses, dim=0)
        return observation

    def val(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_dataset):
                batch_input, batch_target = data
                input_var = batch_input
                target_var = batch_target
                if USE_CUDA:
                    input_var = input_var.cuda()
                    target_var = target_var.cuda()
                    
                batch_output = self.model(input_var)
                # loss = self.criterion(batch_output, target_var)
                acc = accuracy(batch_output, target_var)
                self.val_acc.update(acc)
                self.rewards.append(acc)
        print("Validation Accuracy: {:.4f}".format(acc, self.val_acc.avg))
    
    def get_samples(self):
        try:
            sample = next(self.iter_train_loader)
        except:
            self.iter_train_loader = iter(self.train_loader)
            sample = next(self.iter_train_loader)

        return sample
