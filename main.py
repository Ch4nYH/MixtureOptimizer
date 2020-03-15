import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision
import torchvision.transforms as transforms
from models import SimpleModel, resnet18
from trainer import Trainer
from optimizers import MixtureOptimizer
from meta_train import MetaRunner, MetaTrainer
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.storage import RolloutStorage


import tensorboardX

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default = "")
    parser.add_argument('--epochs', type=int, default = 25)
    parser.add_argument('--optimizer', type = str, default = "mixture")
    parser.add_argument('--length-unroll', type=int, default = 10)
    parser.add_argument('--batch-size', type=int, default = 1000)
    parser.add_argument('--worker', type=int, default = 8)
    parser.add_argument('--dataset', type=str, default = "CIFAR10")
    parser.add_argument('--log-dir', type=str, default = "logs")
    parser.add_argument(
        '--lr-meta', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    args = parser.parse_args()


    task_name = "{}_da{}_ep{}_bs{}".format(args.optimizer, args.dataset, args.epochs, args.batch_size)
    writer = tensorboardX.SummaryWriter(os.path.join(args.log_dir, task_name))

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
    
    if args.dataset == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10('./cifar', transform = data_transforms['train'])
        val_dataset = torchvision.datasets.CIFAR10('./cifar', transform = data_transforms['val'])
    elif args.dataset == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR10('./cifar-100', transform = data_transforms['train'], download = True)
        val_dataset = torchvision.datasets.CIFAR10('./cifar-100', transform = data_transforms['val'], download = True)
        
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=args.worker)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.worker)

    #model = SimpleModel()
    model = resnet18()

    if args.optimizer == 'mixture':
        action_space = np.array([0, 1, 2])
        coord_size = len(model.layers())
        ob_name_lstm = ["loss", "step"]
        ob_name_scalar = []
        num_steps = 3
        obs_shape = (len(ob_name_lstm) + len(ob_name_scalar) + coord_size, )
        _hidden_size = 20
        hidden_size = _hidden_size * len(ob_name_lstm)

        actor_critic = Policy(coord_size, input_size=(len(ob_name_lstm), len(ob_name_scalar)), \
        action_space=len(action_space), hidden_size=_hidden_size, window_size=1)

        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr_meta,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
        meta_info = {'coord_size': coord_size, 'action_space': action_space, 'ob_name_lstm': ob_name_lstm, \
            'ob_name_scalar': ob_name_lstm, 'obs_shape': obs_shape, 'hidden_size': hidden_size, \
            'actor_critic': actor_critic}
    if args.optimizer == 'mixture':
        rollouts = RolloutStorage(num_steps, obs_shape, action_shape=coord_size, hidden_size=hidden_size, num_recurrent_layers=actor_critic.net.num_recurrent_layers)
        names = list(map(lambda x: x[0], list(model.named_parameters())))
        optimizer = MixtureOptimizer(model.parameters(), 0.001, writer = writer, layers = model.layers(), names = names)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 0.001)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 0.001)
    else:
        raise NotImplementedError

    if len(args.gpu) == 0:
        use_cuda = False
    else:
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            use_cuda = True
        except:
            print("Please specify only one GPU id.")
            raise RuntimeError

    trainer = MetaTrainer(model, nn.CrossEntropyLoss(), optimizer, train_loader = train_loader, \
        val_loader = val_loader, USE_CUDA = use_cuda, writer = writer)
    runner = MetaRunner(trainer, rollouts, agent, actor_critic, USE_CUDA = use_cuda, writer = writer)
    runner.run()

if __name__ == '__main__':
    main()