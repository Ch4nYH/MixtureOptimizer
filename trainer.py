from utils import accuracy, AverageMeter
from tqdm import tqdm

class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, USE_CUDA=True, \
        unroll_length = 5, meta = False, val_dataset = None, print_freq = 5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.USE_CUDA = USE_CUDA
        self.unroll_length = unroll_length
        self.stats = {}
        self.meta = meta
        self.val_dataset = val_dataset
        self.train_acc = AverageMeter()
        self.val_acc   = AverageMeter()
        self.print_preq = print_freq
        
        self.reward_baseline = 0
        
        if USE_CUDA:
            self.model = self.model.cuda()
        
    def run(self, epochs=1):
        for i in range(1, epochs + 1):
            self.train_acc.reset()
            self.val_acc.reset()
            print("Epoch [{}/{}]: Training ... ".format(i, epochs))
            self.train()
            self.val()
    
    def reward(self):
        self.reward_baseline = 0.9 * self.reward_baseline + 0.1 * self.val_acc.avg / 100
        return self.val_acc.avg / 100 - self.reward_baseline
    
    def train(self):
        self.model.train()
        for i, data in enumerate(tqdm(self.dataset)):
            batch_input, batch_target = data
            input_var = batch_input
            target_var = batch_target
            if self.USE_CUDA:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            batch_output = self.model(input_var)
            loss = self.criterion(batch_output, target_var)
            acc = accuracy(batch_output, target_var)
            self.train_acc.update(acc)
            if (i % self.print_preq == 0):
                print("Train Accuracy: {}({})".format(acc, self.train_acc.avg))
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.meta and i % self.unroll_length == 0:
                self.val_acc.reset()
                self.val()
                reward = self.reward()
                self.optimizer.meta_update(reward)
        
        if self.meta:
            self.val_acc.reset()
            self.val()
            reward = self.reward()
            self.optimizer.meta_update(reward)
            
        self.iterations += i
    
    def val(self):
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(self.val_dataset):
                batch_input, batch_target = data
                input_var = batch_input
                target_var = batch_target
                if self.USE_CUDA:
                    input_var = input_var.cuda()
                    target_var = target_var.cuda()
                    
                batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var)
                acc = accuracy(batch_output, target_var)
                self.val_acc.update(acc)
            