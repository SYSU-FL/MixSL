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
from utils.options import parse_args
from data.data_utils import get_cifar10, DatasetSplit, random_avg_strategy, get_cifar100
from utils.tools import save_logs, save_cifar100_logs, setup_seed, write_file
from utils.function import random_assign, differential_privacy
from models.resnet import ResNet18_Extractor, ResNet18_Classifer, ResNet34_Extractor, ResNet34_Classifer
from models.vgg import get_split_vgg16
from utils.evaluation import test_inference4split, test_inference4split4cifar100





class Client(object):
    def __init__(self, model, loader, server, args, epoch):
        self.model = model
        self.loader = loader
        self.server = server
        self.args = args
        self.epoch = epoch
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)

    def train(self, weight=None):
        if weight is not None:
            self.model.load_state_dict(weight)
        self.model.train()
        count = 0
        for batch_idx, (images, labels) in enumerate(self.loader):
            self.model.zero_grad()
            if self.args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)
            outputs = self.model(images)
            if args.dp:
                outputs = differential_privacy(outputs, args.gpu)

            loss, grad = self.server.train(outputs, labels)
            outputs.backward(grad)
            self.optimizer.step()
            count += 1
            if self.args.verbose and count == (len(self.loader) - 1):
                print(f"| Data Size: {len(self.loader) * args.local_bs}| loss: {loss.item()}, ")

    
    def get_weight(self,factor=None):
        return self.model.state_dict()

class Server(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        if self.args.gpu > -1:
            self.criterion.cuda(self.args.gpu)
        self.acc_loss = []
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)
        
    def train(self, inputs, labels):
        input_temp = inputs.clone().detach()
        input_temp.requires_grad_()
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(input_temp)
        loss = self.criterion(outputs, labels)
        loss.backward()
        grad = copy.deepcopy(input_temp.grad.data)
        self.optimizer.step()
        return loss, grad

    def get_weight(self,factors=None):
        return self.model.state_dict()

class Trainer(object):
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.trainloader = []
    def _train_val_test(self, dataset, idxs):
        batch_size = args.local_bs
        trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        return trainloader
    def reset(self):
        self.trainloader = []
    def assign(self, idxs_list):
        self.reset()
        for idxs in idxs_list:
            train_loader = self._train_val_test(self.dataset, list(idxs))
            self.trainloader.append(train_loader)

    def train(self, extractor, classifer, epoch):
        index = 1
        server = Server(classifer, self.args)
        #TODO delete the deepcopy
        clients = [Client(copy.deepcopy(extractor), self.trainloader[i], server, self.args, epoch) for i in range(len(self.trainloader))]
        client_w = None
        for client in clients:
            print(f"| Global Round: {epoch} | client index: {index} |")
            client.train(client_w)
            client_w = client.get_weight()
            index += 1
        return client_w, server.get_weight()




if __name__ == "__main__":
    args = parse_args()
    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    TAG = 'sl-' + data_name + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    logs = []
    if args.cifar100:
        train_dataset, test_dataset = get_cifar100(args.balanced)
    else:
        train_dataset, test_dataset = get_cifar10(args.balanced)
    user_groups = random_avg_strategy(train_dataset, args.num_users)
    logs_file = TAG
    client_part = ResNet18_Extractor()
    server_part = ResNet18_Classifer(num_classes)
    #client_part = ResNet34_Extractor()
    #server_part = ResNet34_Classifer(num_classes)
    #client_part, server_part = get_split_vgg16(num_classes)
    if args.gpu > -1:
        client_part.cuda(args.gpu)
        server_part.cuda(args.gpu)

    trainer = Trainer(args, train_dataset)
    for epoch in range(args.epochs):
        local_weights = []
        idxs_users = random_assign(args)
        idx_list = [user_groups[i] for i in idxs_users]
        trainer.assign(idx_list)
        extractor_w, classifer_w = trainer.train(copy.deepcopy(client_part), copy.deepcopy(server_part), epoch)
        client_part.load_state_dict(extractor_w)
        server_part.load_state_dict(classifer_w)
        if args.cifar100:
            test_acc1, test_acc5, test_loss = test_inference4split4cifar100(args, client_part, server_part, test_dataset)
            print("|---- Test Accuracy1: {:.2f}%, Test Accuracy5: {:.2f}%".format(100*test_acc1, 100*test_acc5))
            log_obj = {
                'test_acc1': "{:.2f}%".format(100*test_acc1),
                'test_acc5': "{:.2f}%".format(100*test_acc5),
                'loss': test_loss,
                'epoch': epoch
                }
        else:
            test_acc, test_loss = test_inference4split(args, client_part, server_part, test_dataset)
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            log_obj = {
                    'test_acc': "{:.2f}%".format(100*test_acc),
                    'loss': test_loss,
                    'epoch': epoch
                }   
        logs.append(log_obj)
    if args.cifar100:
        save_cifar100_logs(logs, TAG,  args)
    else:
        save_logs(logs, TAG,  args)

    