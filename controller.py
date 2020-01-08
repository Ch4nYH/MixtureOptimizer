import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils

import numpy as np
import torchvision

from utils import copy_and_pad, preprocess
from optimizers import Optimizer
from collections import defaultdict

class LSTMCoordinator(nn.Module):

    def __init__(self, step_increment = 200, max_length = 0, hidden_layers = 2, hidden_units = 20, input_dim = 2, \
        num_actions = 3):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.input_dim = input_dim

        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_units, self.hidden_layers)
        self.max_length = max_length
        self.decoder = nn.Linear(2 * self.hidden_units, num_actions)
    def forward(self, x, hx, cx):
        
        if (hx is None):
            output, (hx, cx) = self.lstm(x)
        else:
            output, (hx, cx) = self.lstm(x, (hx, cx))

        lstm_out, _ = rnnutils.pad_packed_sequence(output, batch_first = True)
        lstm_out = hx.reshape(lstm_out.shape[0], -1)
        logit = self.decoder(lstm_out)
        prob = torch.nn.functional.softmax(logit, dim=-1)

        action = prob.multinomial(1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
        selected_log_probs = log_prob.gather(1, action.data)
        return action, selected_log_probs, hx, cx    

class MixtureOptimizer(object):
    def __init__(self, parameters, meta_alpha, coordinator = LSTMCoordinator, meta_optimizer = optim.Adam, length_unroll = 20,\
        alpha = 0.001, beta1 = 0.9, beta2 = 0.999, eta = 1e-8, USE_CUDA = True):
        param = list(parameters)
        self.parameters = param
        
        max_length = max(list(map(lambda x:x.view(1,-1).shape[1], self.parameters)))
        self.coordinator = coordinator(max_length = max_length)
        self.meta_optimizer = meta_optimizer(self.coordinator.parameters(), lr = meta_alpha)
        self.length_unroll = length_unroll
        self.hx = None
        self.cx = None
        self.meta_step = 0
        self.selected_log_probs = []

        
        
        self.state = defaultdict(dict)
        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        
        self.update_rules = [lambda x, para: self.alpha * x, \
            lambda x, para: self.alpha / (self.eta + torch.sqrt(self.state[para]['r'])) * x, \
            lambda x, para: self.alpha * self.state[para]['mt_hat'] / (torch.sqrt(self.state[para]['vt_hat']) + self.eta)]
        
        self.USE_CUDA = USE_CUDA
        if self.USE_CUDA:
            self.coordinator = self.coordinator.cuda()
        
    def reset(self):
        self.hx = None
        self.cx = None
        self.selected_log_probs = []

    def meta_act(self, gradients, length_gradients):
        inputs = preprocess(gradients)
        packed_inputs = rnnutils.pack_padded_sequence(inputs, length_gradients, batch_first = True, enforce_sorted = False)
        action, selected_log_probs, self.hx, self.cx = self.coordinator(packed_inputs, self.hx, self.cx)

        self.action = action.cpu().numpy().flatten()
        self.selected_log_probs.append(selected_log_probs.flatten())
        self.meta_step += 1
    def meta_update(self, rewards):
        self.selected_log_probs = torch.cat(self.selected_log_probs, 0)
        action_loss = -sum(self.selected_log_probs) * rewards
        self.meta_optimizer.zero_grad()
        action_loss.backward()
        self.meta_optimizer.step()
        self.selected_log_probs = []
        self.reset()
        
    def step(self):
        gradients = list(map(lambda x: x.grad.data.detach().view(1, -1), self.parameters))
        padded_grad, length_grad = copy_and_pad(gradients)
        
        self.meta_act(padded_grad, length_grad)
        count = 0
        for p in self.parameters:
            state = self.state[p]
            if len(state) == 0: # State initialization
                state['t'] = 0
                state['mt'] = 0
                state['vt'] = 0
                state['mt_hat'] = 0
                state['vt_hat'] = 0
                state['r'] = 0
                
            grad = p.grad.data
            state['t'] = state['t'] + 1
            state['mt'] = self.beta1 * state['mt'] + (1 - self.beta1) * grad
            state['vt'] = self.beta2 * state['vt'] + (1 - self.beta2) * (grad ** 2)
            state['mt_hat'] = state['mt'] / (1 - np.power(self.beta1, state['t']))
            state['vt_hat'] = state['vt'] / (1 - np.power(self.beta2, state['t']))
            state['r'] = state['r'] + (grad ** 2)
            p.data.add_(-self.update_rules[self.action[count]](grad, p))
        
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    

if __name__ == '__main__':
    
    x = torch.randn((1, 3, 224, 224))
    label = torch.zeros(1).long()
    model = torchvision.models.AlexNet(2)
    
    criterion = nn.CrossEntropyLoss()
    controller = MixtureOptimizer(model.parameters(), 0.001)
    for i in range(10):
        y = model(x)
        loss = criterion(y, label)
        
        controller.zero_grad()
        loss.backward()
        controller.step()
        print(loss)