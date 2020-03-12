import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
class SimpleModel(nn.Module):
    def __init__(self, num_classes = 10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.l1 = nn.Linear(1024, 128)
        self.l2 = nn.Linear(128, num_classes)

    def layers(self):
        return ['conv1', 'bn1', 'conv2', 'bn2', 'l1', 'l2']

    def reset(self):
        def init_(m):
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data,0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()
        self.apply(init_)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x     


models = {'simple': SimpleModel}
def get_model(name):
    if name in models:
        return name()
    else:
        raise NotImplementedError