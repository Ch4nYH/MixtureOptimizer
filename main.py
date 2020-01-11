import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import torchvision
import torchvision.transforms as transforms
from controller import MixtureOptimizer
from trainer import Trainer

import tensorboardX
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action="store_true")
parser.add_argument('--task', type=str)
parser.add_argument('--epochs', type=int, default = 20)
parser.add_argument('--meta', action="store_true")
parser.add_argument('--length-unroll', type=int)
parser.add_argument('--batch-size', type=int, default = 256)
parser.add_argument('--worker', type=int, default = 8)
args = parser.parse_args()
writer = tensorboardX.SummaryWriter('logs/' + args.task)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
} 

train_dataset = torchvision.datasets.CIFAR10('./cifar', transform = data_transforms['train'])
val_dataset = torchvision.datasets.CIFAR10('./cifar', transform = data_transforms['val'])

train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=args.worker)
val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.worker)



class Model(nn.Module):
    def __init__(self, num_classes = 10):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size = 3, stride = 2, padding = 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.l1 = nn.Linear(1024, 128)
        self.l2 = nn.Linear(128, num_classes)
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
model = Model()
if args.meta:
    optimizer = MixtureOptimizer(model.parameters(), 0.001, writer = writer)
else:
    optimizer = torch.optim.Adagrad(model.parameters(), 0.001)
trainer = Trainer(model, nn.CrossEntropyLoss(), optimizer = optimizer, dataset = train_loader, \
    val_dataset=val_loader, USE_CUDA = not args.no_cuda, meta = args.meta, writer = writer, unroll_length=args.length_unroll)
trainer.run(epochs = args.epochs)