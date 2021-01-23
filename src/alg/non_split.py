import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import argparse
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from tools.options import parse_args
from dataprocess.data_utils import get_cifar10, get_cifar100
from tools.utils import save_logs, save_cifar100_logs, setup_seed
from tools.function import random_assign
from model.resnet import ResNet18, ResNet34
from model.vgg import get_vgg16
from tools.evaluation import test_inference, test_inference4cifar100 





if __name__ == "__main__":
    args = parse_args()
    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    split_type = 'googlesplit' if args.google_split else ''
    TAG = 'non-split-' + data_name + '-' + args.name + '-' + split_type + '-'
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    logs = []
    if args.cifar100:
        train_dataset, test_dataset = get_cifar100(args.balanced)
    else:
        train_dataset, test_dataset = get_cifar10(args.balanced)
    
    #net = ResNet18(num_classes)
    #net = ResNet34(num_classes)
    net = get_vgg16(num_classes)
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        net.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
    trainloader = DataLoader(train_dataset, batch_size=args.local_bs, shuffle=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    net.train()
    
    for epoch in range(args.epochs):
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(args.gpu), labels.to(args.gpu)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            if batch_idx % 1000 == 0:
                print(f"epoch: {epoch}, Loss: {loss}")
            loss.backward()
            optimizer.step()
        
        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4cifar100(args, net, test_dataset)
            print(f"|----Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
            log_obj = {
                'test_acc1': "{:.2f}%".format(100*test_acc1),
                'test_acc5': "{:.2f}%".format(100*test_acc5),
                'loss': test_loss,
                'epoch': epoch 
                }
            logs.append(log_obj)
        else:
            test_acc, test_loss = test_inference(args, net, test_dataset)
            print(f"|----Test Loss: {test_loss}, Test Accuracy: {100*test_acc}%")
            log_obj = {
                'test_acc': "{:.2f}%".format(100*test_acc),
                'loss': test_loss,
                'epoch': epoch 
                }
            logs.append(log_obj)
    if args.cifar100:
        save_cifar100_logs(logs, TAG, args)
    else:
        save_logs(logs, TAG,  args)

    