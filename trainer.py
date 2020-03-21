import torch
from utils import accuracy, AverageMeter
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from collections import deque

import numpy as np


class Trainer(object):

    def __init__(self, model, criterion, optimizer, USE_CUDA=True, \
        train_loader = None, val_loader = None, print_freq = 5, writer = None, epochs = 30):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.USE_CUDA = USE_CUDA

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_acc = AverageMeter()
        self.val_acc   = AverageMeter()
        self.print_preq = print_freq

        self.writer = writer
        self.window_size = 5

        self.iter_train_loader = iter(self.train_loader)
        self.iter_val_loader = iter(self.val_loader)

        self.total_steps = 30 * len(self.train_loader)
        self.total_steps_epoch = len(self.train_loader)
        self.total_steps_val = len(self.val_loader)
        self.step = 0
        self.epochs = epochs
        if USE_CUDA:
            self.model = self.model.cuda()

    def get_steps(self):
        return self.epochs * self.total_steps,self.total_steps_epoch

    def observe(self):
        losses = []
        for idx in range(self.window_size):
            train_loss, train_acc = self.train_step()
            losses.append(train_loss.detach())

        losses = [sum(losses) / len(losses)]
        return torch.tensor(losses)

    def get_steps(self):
        return self.total_steps,self.total_steps_epoch
    def reset(self):
        self.step = 0
        self.model.reset()

    def train_step(self): 
        self.model.train()
        self.optimizer.zero_grad()
        input, label = self.get_train_samples()
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
        self.model.val()
        losses = []
        with torch.no_grad():
            input, label = self.get_val_samples()
            if self.USE_CUDA:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            acc = accuracy(output, label)
            loss = self.criterion(output, label.long())

        return loss

    def val(self):
        self.model.val()
        accs = []
        losses = []
        with torch.no_grad():
            for _ in range(self.total_steps_val): 
                input, label = self.get_val_samples()
                if self.USE_CUDA:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                acc = accuracy(output, label)
                loss = self.criterion(output, label.long())
                losses.append(loss)
                accs.append(acc)
        return np.mean(acc), np.mean(losses)

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

class Runner(object):
    def __init__(self, trainer, meta_epochs = 50, USE_CUDA = False, writer = None):
        self.trainer = trainer
        self.meta_epochs = meta_epochs
        self.total_steps, self.total_steps_epoch = trainer.get_steps()
        self.step = 0
        self.window_size = self.trainer.window_size
        self.USE_CUDA = USE_CUDA
        self.writer = writer

        self.layers = self.trainer.model.layers()
        self.meta_epochs = meta_epochs
        self.use_gae = True
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.accumulated_step = 0

    def reset(self):
        self.trainer.reset()

    def run(self):
        for idx in range(self.meta_epochs):
            self.reset()
            self.step_run(idx)
    def step_run(self, epoch):
        prev_loss = self.trainer.observe()
        self.step += self.window_size
        while self.step < self.total_steps:
            self.step += self.window_size
            curr_loss = self.trainer.observe()

            if self.step % self.total_steps_epoch == 0:
                acc, loss = self.trainer.val()
                self.writer.add_scalar("val/acc", acc, self.step + self.accumulated_step)
                self.writer.add_scalar("val/loss", loss, self.step + self.accumulated_step)