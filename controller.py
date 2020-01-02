import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils

import numpy as np

from utils import *
from optimizers import Optimizer

class LSTMCoordinator(nn.Module):

    def __init__(self, step_increment = 200, max_length = 0, hidden_layers = 2, hidden_units = 20, input_dim = 2, num_actions = 3):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.input_dim = input_dim

        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_units, self.hidden_layers)
        self.max_length = max_length
        self.decoder = nn.Linear(self.max_length * self.hidden_units, num_actions)
    def forward(self, x, hx, cx):
        if (hx is None):
            hx, cx = self.lstm(x)
        else:
            hx, cx = self.lstm(x, (hx, cx))

        lstm_out, _ = rnnutils.pad_packed_sequence(hx, batch_first = True)
        lstm_out = lstm_out.reshape(lstm_out.shape[0], -1)
        logit = self.decoder(lstm_out)
        prob = torch.nn.functional.softmax(logit, dim=-1)

        action = prob.multinomial(1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
        selected_log_probs = log_prob.gather(1, action.data)

        return action, selected_log_probs, hx, cx    

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
    def reset(self):
        self.hx = None
        self.cx = None
        self.selected_log_probs = []

    def meta_update(self, gradients, length_gradients):
        inputs = preprocess(gradients)
        print(inputs.shape)
        packed_inputs = rnnutils.pack_padded_sequence(inputs, length_gradients, batch_first = True)
        action, selected_log_probs, self.hx, self.cx = self.coordinator(packed_inputs, self.hx, self.cx)

        self.action = action.numpy()
        self.selected_log_probs.append(selected_log_probs)
        self.step += 1

    def perform_update(self, parameters, gradients, names):
        num_parameters = len(self.action)
        for i in range(num_parameters):
            print(parameters[i])
            print(gradients[i])
            print(names[i])
            print(self.action[i])
            self.optimizers(parameters[i], gradients[i], names[i], self.action[i])

if __name__ == '__main__':
    
    x = torch.randn((10, 20))
    model = nn.Linear(20, 1)
    y = model(x)
    loss = torch.sum(y ** 2)
    loss.backward()
    para, grad, names = get_parameters_and_gradients(model)
    print(names)
    padded_para, length_para = copy_and_pad(para)
    padded_grad, length_grad = copy_and_pad(grad)
    controller = Controller(LSTMCoordinator(max_length = length_grad[0]), optim.Adam, optimizers = Optimizer())
    controller.meta_update(padded_grad, length_grad)
    controller.perform_update(para, grad, names)