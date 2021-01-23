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
from dataprocess.data_utils import get_cifar10, DatasetSplit, random_avg_strategy, get_cifar100
from tools.utils import save_logs, save_cifar100_logs, setup_seed
from tools.function import random_assign, assgin_task
from tools.evaluation import test_inference4split, test_inference4split4cifar100



class Client(object):
    def __init__(self, model, loader, server, args, num_per_cls):
        self.model = model
        self.loader = loader
        self.server = server
        self.args = args
        self.num_per_cls = num_per_cls
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)

    def train(self, weight=None):
        if weight is not None:
            self.model.load_state_dict(weight)
        self.model.train()
        count = 0
        for batch_idx, (images, labels) in enumerate(self.loader):
            self.model.zero_grad()
            if self.args.gpu > -1:
                images, labels = images.cuda(self.args.gpu), labels.cuda(self.args.gpu)
            outputs = self.model(images)
            outputs, label_x, label_y, label_z, lam, lam2, lam_y, lam_y2 = self._triple_mixup(outputs, labels, 1.0, False)
            loss, grad = self.server.train(outputs, label_x, label_y, label_z, lam_y, lam_y2)
            outputs.backward(grad)
            self.optimizer.step()
            count += 1
            if self.args.verbose and count == (len(self.loader) - 1):
                print(f"| Data Size: {len(self.loader) * args.local_bs}| loss: {loss.item()}, ")

    
    def get_weight(self,factor=None):
        return self.model.state_dict()
    
    def _mixup_data(self, x, y, alpha=1.0, balanced=True):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda(self.args.gpu)
        
        index = index.cuda(self.args.gpu)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        y_lam = None
        if not balanced:
            y_lam = []
            for i in range(y_a.size(0)):
                y_elem = self._get_imbalanced_factor(y_a[i], y_b[i], lam)
                y_lam.append(y_elem)
        return mixed_x, y_a, y_b, lam, y_lam


    def _get_imbalanced_factor(self, a, b, x_factor, k=2, lam=0.5):
        y_factor = 0.
        if (self.num_per_cls[a] / self.num_per_cls[b] >= k) and (x_factor < lam):
            y_factor = 0.
        elif (self.num_per_cls[a] / self.num_per_cls[b] <= 1) and ((1 - x_factor) < lam):
            y_factor = 1.
        else:
            y_factor = x_factor
        return y_factor
    

    def _triple_mixup(self, x, y, alpha=1.0, balanced=True):
        if alpha > 0:
            lam1 = np.random.beta(alpha, alpha)
            lam2 = np.random.beta(alpha, alpha)
        else:
            lam1 = 1
            lam2 = 1
        batch_size = x.size()[0]
        index1 = torch.randperm(batch_size).cuda(self.args.gpu)
        index2 = torch.randperm(batch_size).cuda(self.args.gpu)
        mixed_x = lam2 * (lam1 * x + (1 - lam1) * x[index1, :]) + (1 - lam2) * x[index2, :]
        y_a, y_b, y_c = y, y[index1], y[index2]
        y_lam = lam1
        y_lam2 = lam2
        if not balanced:
            y_lam = []
            y_lam2 = []
            for i in range(y_a.size(0)):
                y_elem1, y_elem2 = self._get_imbalanced_triple_factor(y_a[i], y_b[i], y_c[i], lam1, lam2, self.args.k, self.args.pai)
                y_lam.append(y_elem1)
                y_lam2.append(y_elem2)
        return mixed_x, y_a, y_b, y_c, lam1, lam2, y_lam, y_lam2
        
                

    def _get_imbalanced_triple_factor(self, a, b, c, x_factor, x_factor2, k=2, lam=0.5):
        y_factor = 0.
        y_factor2 = 0.
        tag = -1
        if (self.num_per_cls[a] / self.num_per_cls[b] >= k) and (x_factor < lam):
            y_factor = 0.
            tag = b
        elif (self.num_per_cls[a] / self.num_per_cls[b] <= 1) and ((1 - x_factor) < lam):
            y_factor = 1.
            tag = a
        else:
            y_factor = x_factor
            tag = a if y_factor > 0.5 else b
        
        if tag == a:
            if (self.num_per_cls[a] / self.num_per_cls[c] >= k) and (x_factor2 < lam):
                y_factor2 = 0.
            elif (self.num_per_cls[a] / self.num_per_cls[c] <= 1) and ((1 - x_factor2) < lam):
                y_factor2 = 1.
            else:
                y_factor2 = x_factor2
        else:
            if (self.num_per_cls[b] / self.num_per_cls[c] >= k) and (x_factor2 < lam):
                y_factor2 = 0.
            elif (self.num_per_cls[b] / self.num_per_cls[c] <= 1) and ((1 - x_factor2) < lam):
                y_factor2 = 1.
            else:
                y_factor2 = x_factor2
        return y_factor, y_factor2





    

    
    
    


    


class Server(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        if self.args.gpu > -1:
            self.criterion.cuda(self.args.gpu)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)
        

    def train(self, inputs, labels0, labels1, labels2, lam, lam2): ##
        input_temp = inputs.clone().detach()
        input_temp.requires_grad_()
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(input_temp)
        lam_y = torch.FloatTensor(lam).cuda(self.args.gpu)
        lam_y2 = torch.FloatTensor(lam2).cuda(self.args.gpu)
        one_vector = torch.ones(lam_y.size()).cuda(self.args.gpu)
        one_vector2 = torch.ones(lam_y2.size()).cuda(self.args.gpu)
        loss = (lam_y2 * (lam_y * self.criterion(outputs, labels0) + (one_vector - lam_y) * self.criterion(outputs, labels1)) + (one_vector2 - lam_y2) * self.criterion(outputs, labels2)).sum()
        loss /= outputs.size(0)
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
        self.cls_num_list = []
    def _train_val_test(self, dataset, idxs):
        batch_size = args.local_bs
        trainloader = DataLoader(DatasetSplit(dataset, idxs), batch_size=batch_size, shuffle=True)
        return trainloader
    def reset(self):
        self.trainloader = []
        self.cls_num_list = []
    def assign(self, idxs_list, cls_num_list):
        self.reset()
        self.cls_num_list = cls_num_list
    
        for idxs in idxs_list:
            train_loader = self._train_val_test(self.dataset, list(idxs))
            self.trainloader.append(train_loader)

    def train(self, extractor, classifer, epoch):
        index = 1
        server = Server(classifer, self.args)
        #TODO delete the deepcopy
        clients = [Client(copy.deepcopy(extractor), self.trainloader[i], server, self.args, self.cls_num_list[i]) for i in range(len(self.trainloader))]
        client_w = None
        for client in clients:
            print(f"| Global Round: {epoch} | client index: {index} |")
            client.train(client_w)
            client_w = client.get_weight()
            index += 1
        return client_w, server.get_weight()




if __name__ == "__main__":
    args = parse_args()
    client_part, server_part, train_dataset, test_dataset,\
    user_groups, cls_num_per_clients, prefix_tag = assgin_task(args)
    TAG = '3-labels-mixsl-' + prefix_tag + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    logs = []

    trainer = Trainer(args, train_dataset)
    for epoch in range(args.epochs):
        local_weights = []
        idxs_users = random_assign(args)
        idx_list = [user_groups[i] for i in idxs_users]
        cls_num_list = [list(cls_num_per_clients[i]) for i in idxs_users]
        trainer.assign(idx_list, cls_num_list)
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
    