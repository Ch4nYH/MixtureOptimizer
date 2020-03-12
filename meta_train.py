import torch
from utils import accuracy, AverageMeter
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage



class MetaTrainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, USE_CUDA=True, \
        train_loader = None, val_loader = None, print_freq = 5, writer = None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.USE_CUDA = USE_CUDA

        self.train_loader = train_loader
        self.val_loader = val_dataset

        self.train_acc = AverageMeter()
        self.val_acc   = AverageMeter()
        self.print_preq = print_freq

        self.writer = writer
        self.window_size = 5

        self.iter_train_loader = iter(self.train_loader)
        self.iter_val_loader = iter(self.val_loader)

        self.total_steps = 30 * len(self.train_loader)
        self.total_steps_epoch = len(self.train_loader)

        if USE_CUDA:
            self.model = self.model.cuda()

    def get_steps(self):
        return self.total_steps,self.total_steps_epoch
    
    def observe(self):
        losses = []
        optimizee_step = []
        for idx in range(self.window_size):
            train_loss, train_acc = self.train_step()
            losses.append(train_loss.detach())
            optimizee_step.append((self.step + idx) / self.total_steps_epoch)

        losses = [sum(losses) / len(losses)]
        optimizee_step = [sum(optimizee_step) / len(optimizee_step)]
        optimizee_step = [torch.tensor(step).cuda() for step in optimizee_step]
        observation = torch.stack(losses + optimizee_step, dim=0)
        return observation, torch.tensor(losses)

    def train_step(self): 
       
        self.optimizer.zero_grad()
        input, label = self.get_samples()
        if self.USE_CUDA:
            label = label.cuda()
            input = input.cuda()

        output = self.model(input)
        loss = self.criterion(output, label.long())
        acc = accuracy(output, label)
        loss.backward()
        self.optimizer.step()
        return loss, acc
            
    def val_step(self):
        losses = []
        with torch.no_grad():
            input, label = self.get_samples()
            if self.USE_CUDA:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            acc = accuracy(output, label)
            loss = self.criterion(output, label.long())

        return loss

    def train_val_step(self):
        pass

    def get_train_samples(self):
        try:
            sample = next(self.iter_train_loader)
        except:
            self.iter_train_loader = iter(self.train_loader)
            sample = next(self.iter_train_loader)

        return sample

    def get_val_samples(self):
        try:
            sample = next(self.iter_val_loader)
        except:
            self.iter_val_loader = iter(self.val_loader)
            sample = next(self.iter_val_loader)

        return sample
    def get_optimizer(self):
        return self.optimizer

class MetaRunner(object):
    def __init__(self, trainer, rollouts, agent, ac, num_steps = 3, meta_epochs = 50, USE_CUDA = False, writer = None):
        self.trainer = trainer
        self.rollouts = rollouts
        self.agent = agent
        self.ac = ac
        self.num_steps = num_steps
        self.meta_epochs = meta_epochs
        self.total_steps_epoch, self.total_steps = trainer.get_steps()
        self.step = 0
        self.window_size = self.trainer.window_size
        self.USE_CUDA = USE_CUDA
        self.num_steps = num_steps
        self.writer = writer

        self.layers = self.trainer.model.layers()
        self.meta_epochs = meta_epochs

    def run(self):
        for idx in range(self.meta_epochs):
            self.step(idx)
    def step(self, epoch):
        observation, prev_loss = self.trainer.observe()
        self.step += self.window_size
        rollouts.obs[0].copy_(observation)
        episode_rewards = deque(maxlen=100)
        while self.step < self.total_steps_epoch:
            for step in range(self.num_steps):
                with torch.no_grad():
                    self.step += self.window_size
                    value, action, action_log_prob, recurrent_hidden_states, distribution = \
                    self.ac.act(rollouts.obs[step:step+1], self.rollouts.recurrent_hidden_states[step])
                    action = action.squeeze(0)
                    action_log_prob = action_log_prob.squeeze(0)
                    value = value.squeeze(0)
                    for idx in range(len(action)):
                        writer.add_scalar("action/%s"%self.layers[idx], action[idx], self.step)
                        writer.add_scalar("entropy/%s"%self.layers[idx], distribution.distributions[idx].entropy(), self.step)
                    self.trainer.get_optimizer().set_actions(action.numpy())
                observation, curr_loss = self.trainer.observe()
                reward = curr_loss - prev_loss
                episode_rewards.append(float(reward.cpu().numpy()))
                self.writer.add_scalar("reward", reward, self.step)
                self.rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward)

            with torch.no_grad():
                next_value = self.ac.get_value(rollouts.obs[-1:], rollouts.recurrent_hidden_states[-1]).detach()
            self.rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda)
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            
            writer.add_scalar("value_loss", value_loss, self.step)
            writer.add_scalar("action_loss", action_loss, self.step)

            print("action_loss:", action_loss, " @ %d steps"%(current_optimizee_step))

            rollouts.after_update()


