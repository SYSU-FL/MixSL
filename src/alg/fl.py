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
from dataprocess.data_utils import get_cifar10, DatasetSplit, random_avg_strategy, get_cifar100, cifar_noniid
from tools.utils import save_logs, save_cifar100_logs, setup_seed
from tools.function import random_assign, assgin_task
from tools.evaluation import test_inference, test_inference4cifar100
from model.resnet import ResNet18, ResNet34
from model.vgg import get_vgg16

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class Client(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss().cuda(self.args.gpu)

    def _train_val_test(self, dataset, idxs):
        trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                 batch_size=self.args.local_bs, shuffle=True)
        return trainloader, None, None

    def assign(self, idxs):
        self.trainloader, self.validloader, self.testloader = self._train_val_test(self.dataset, list(idxs))
        return len(self.trainloader)

    def train(self, model, epoch):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)
        for e in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx == (len(self.trainloader) - 1):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, e, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
        return model.state_dict()

        



if __name__ == '__main__':
    #init...
    args = parse_args()
    _, _, train_dataset, test_dataset,\
    user_groups, cls_num_per_clients, prefix_tag = assgin_task(args)
    TAG = 'fedavg-' + prefix_tag + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    logs = []
    # simulate clients
    local_model = Client(args=args, dataset=train_dataset)
    num_class = 100 if args.cifar100 else 10
    global_model = ResNet18(num_class)
    if args.gpu > -1:
        global_model.cuda(args.gpu)
    
    for epoch in range(args.epochs):
        count = 0
        local_weights = []
        print(f'epoch: {epoch}')
        global_model.train()
        global_weights = global_model.state_dict()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            count += 1
            print(f'Epoch: {epoch}-count: {count}-id:{idx}')
            local_model.assign(user_groups[idx])
            local_weights.append(local_model.train(copy.deepcopy(global_model), epoch))
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4cifar100(args, global_model, test_dataset)
            print(f"|----Test Loss: {test_loss}, Test Accuracy_correct1: {100*test_acc1}%, Test Accuracy_correct5: {100*test_acc5}%")
            log_obj = {
                'test_acc1': "{:.2f}%".format(100*test_acc1),
                'test_acc5': "{:.2f}%".format(100*test_acc5),
                'loss': test_loss,
                'epoch': epoch 
               
                }
            logs.append(log_obj)
        else:
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
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

            


