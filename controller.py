import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from utils import *
from optimizers import AdaGrad, SGD, Adam

class LSTMCoordinator(nn.Module):

    def __init__(self, step_increment = 200):
        super().__init__()

        self.lstm = torch.nn.LSTMCell(2, 20, 2)
        self.decoder = nn.Linear(20, 3)

    def forward(self, x, hx, cx):
        if (hx is None):
            hx, cx = self.lstm(x)
        else:
            hx, cx = self.lstm(x, (hx, cx))

        logit = self.decoder(hx)
        prob = torch.nn.functional.softmax(logit, dim=-1)

        action = prob.multinomial(1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
        selected_log_probs = log_prob.gather(1, action.data)

        return action, selected_log_probs, hx, cx    

    def get_selected_log_probs(self):
        selected_log_probs = self.selected_log_probs
        self.selected_log_probs = []
        return selected_log_probs

class Controller(object):
    def __init__(self, coordinator = None, optimizer = None, alpha = 0.001, length_unroll = 20, optimizers = []):
        self.coordinator = coordinator
        self.optimizer = optimizer(self.coordinator.parameters(), lr = alpha)
        self.length_unroll = length_unroll
        self.hx = None
        self.cx = None
        self.step = 0

        self.selected_log_probs = []
        self.optimizers = optimizers
        self.num_optimizers = len(self.optimizers)
    def reset(self):
        self.hx = None
        self.cx = None
        self.selected_log_probs = []

    def meta_update(self, gradients):
        inputs = preprocess(gradients)
        action, selected_log_probs, self.hx, self.cx = self.controller(inputs)

        self.action = action
        self.selected_log_probs.append(selected_log_probs)
        self.step += 1

    def perform_update(self, parameters, gradients):
        for i in range(self.num_optimizers):
            parameters_i = torch.masked_select(parameters, self.action == i)
            gradients_i  = torch.masked_select(gradients,  self.action == i)
            self.optimizers[i](parameters_i, gradients_i)


if __name__ == '__main__':
    controller = LSTMCoordinator()
    x = torch.randn((10, 2))
    model = nn.Linear(2, 1)
    y = model(x)
    loss = torch.sum(y ** 2)
    loss.backward()
    para, grad = get_parameters_and_gradients(model)
    print(para)
    print(grad)
    sgd = SGD()
    sgd(para, grad)
    print(para)
    print(grad)