import torch
import numpy as np

def preprocess(x, k = 5):
    log = torch.log(torch.abs(x) + 1e-7)
    clamped_log = (log / k).clamp(min = -1)
    sign = (x * np.exp(k)).clamp(min = -1, max = 1)
    return torch.cat([clamped_log.unsqueeze(-1), sign.unsqueeze(-1)], dim = -1)

def get_parameters_and_gradients(model)->list:
    parameters = list(map(lambda x: x.data.view(1, -1), model.parameters()))
    gradients = list(map(lambda x: x.data.view(1, -1), map(lambda x: x.grad, model.parameters())))
    parameters_name = list(map(lambda x: x[0], model.named_parameters()))
    return parameters, gradients, parameters_name

def copy_and_pad(tensors: list):
    real_lengths = list(map(lambda x: x.shape[1], tensors))
    max_length = max(real_lengths)
    def pad(tensor, l):
        new_tensor = torch.zeros((1, l), device = tensor.device)
        new_tensor[0, :tensor.shape[1]] = tensor
        return new_tensor
    new_tensors = map(lambda x: pad(x.clone(), max_length), tensors)

    return torch.cat(list(new_tensors), 0), real_lengths

def update_parameter(parameter, delta):
    parameter.add_(delta)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0].item()
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    l = [torch.randn((1, 2)), torch.zeros((1,3))]
    print(copy_and_pad(l))