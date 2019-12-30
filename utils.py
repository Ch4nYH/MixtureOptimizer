import torch
import numpy as np

def preprocess(x, k = 5):
    log = torch.log(torch.abs(x) + 1e-7)
    clamped_log = (log / k).clamp(min = -1)
    sign = (x * np.exp(k)).clamp(min = -1, max = 1)
    return torch.cat([clamped_log, sign], axis = 1)

def get_parameters_and_gradients(model):
    parameters = list(map(lambda x: x.data.view(-1, 1), model.parameters()))
    gradients = list(map(lambda x: x.data.view(-1, 1), map(lambda x: x.grad, model.parameters())))
    return torch.cat(parameters, 0), torch.cat(gradients, 0)

def update_parameter(parameter, delta):
    parameter.add_(delta)