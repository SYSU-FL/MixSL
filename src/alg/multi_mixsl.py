import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time
import argparse
import math
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from tools.options import parse_args
from dataprocess.data_utils import get_cifar10, DatasetSplit, random_avg_strategy, get_cifar100, cifar_noniid
from tools.utils import save_logs, save_cifar100_logs, setup_seed, write_file
from tools.function import random_assign
from tools.evaluation import test_inference4split, test_inference4split4cifar100



class Client(object):
    def __init__(self, model, loader, server, args, num_per_cls, epoch):
        self.model = model
        self.loader = loader
        self.server = server
        self.args = args
        self.num_per_cls = num_per_cls
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)
        self.epoch = epoch

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
            outputs, labels_vectors, lam_yss = self._mixup(outputs, labels, self.args.mix_num)
            loss, grad = self.server.train(outputs, labels_vectors, lam_yss)
            outputs.backward(grad)
            self.optimizer.step()
            count += 1
            if self.args.verbose and count == (len(self.loader) - 1):
                print(f"| Data Size: {len(self.loader) * args.local_bs}| loss: {loss.item()}, ")

    def get_weight(self,factor=None):
        return self.model.state_dict()
    
    
    def _mixup(self, x, y, num=3):
        x_ = x
        y_ = y
        lam_yss = []
        label_tuples = []
        for i in range(num-1):
            mixed_x, y_a, y_b, lam, lam_ys = self._mixup_data(x_, y_)
            lam_yss.append(lam_ys)
            if i == 0:
                label_tuples.append(y_a)
            label_tuples.append(y_b)
            x_ = mixed_x
            y_ = self._max_labels(y_a, y_b, lam_ys)
        return mixed_x, label_tuples, lam_yss


    def _max_labels(self, y_a, y_b, lam_ys):
        max_labs = []
        for i in range(len(lam_ys)):
            max_labs.append(y_a[i] if lam_ys[i] > 0.5 else y_b[i])
        return torch.Tensor(max_labs).type_as(y_a).view(y_a.size()).cuda(self.args.gpu)

    def _mixup_data(self, x, y):
        lam = np.random.beta(1., 1.)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        lam_ys = []
        for i in range(y_a.size(0)):
            lam_ys.append(self.get_labels_factors(y_a[i], y_b[i], lam))
        return mixed_x, y_a, y_b, lam, lam_ys


    def get_labels_factors(self, a, b, lam, k=2, pai=0.5):
        lam_y = 0.
        if (self.num_per_cls[a] / self.num_per_cls[b] >= k) and (lam < pai):
            lam_y = 0.
        elif (self.num_per_cls[a] / self.num_per_cls[b] <= 1) and ((1 - lam) < pai):
            lam_y = 1.
        else:
            lam_y = lam
        return lam_y
   

class Server(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        if self.args.gpu > -1:
            self.criterion.cuda(self.args.gpu)
            self.softmax.cuda(self.args.gpu)
        self.acc_loss = []
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=1e-5)
        self.inputs = []
    

    def train(self, inputs, labels_vectors, lam_yss): ##
        input_temp = inputs.clone().detach()
        input_temp.requires_grad_()
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(input_temp)
        loss = self._loss(outputs, labels_vectors, lam_yss)
        loss.backward()
        grad = copy.deepcopy(input_temp.grad.data)
        self.optimizer.step()
        return loss, grad

    def _loss(self, outputs, labels_vectors, lam_yss):
        lam_y = torch.FloatTensor(lam_yss[0]).cuda(self.args.gpu)
        one_vector = torch.ones(lam_y.size()).cuda(self.args.gpu)
        loss = (lam_y * self.criterion(outputs, labels_vectors[0])) + \
        ((one_vector - lam_y) * self.criterion(outputs, labels_vectors[1]))
        for i in range(1, len(lam_yss)):
            lam_y = torch.FloatTensor(lam_yss[i]).cuda(self.args.gpu)
            one_vector = torch.ones(lam_y.size()).cuda(self.args.gpu)
            loss = (lam_y * loss) + ((one_vector - lam_y) * self.criterion(outputs, labels_vectors[i + 1]))
        loss = loss.sum()
        loss /= outputs.size(0)
        return loss
            
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
        clients = [Client(copy.deepcopy(extractor), self.trainloader[i], server, self.args, self.cls_num_list[i], epoch) for i in range(len(self.trainloader))]
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

    TAG = 'multi-mixsl-mixsum' + str(args.mix_num) + '-' + prefix_tag + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    logs = []

    start_epoch = 0
    start_acc = 0.
    if args.resume:
        print("===> resuming from checkpoint...")
        if not os.path.exists('../checkpoints'):
            os.mkdir('../checkpoints')
        if os.path.exists('../checkpoints/' + args.reload_path + '_ckpt'):
            ckpt = torch.load('../checkpoints/' + args.reload_path + '_ckpt')
            client_part.load_state_dict(ckpt['client'])
            server_part.load_state_dict(ckpt['server'])
            start_epoch = int(ckpt['epoch'])
            start_acc = ckpt['acc']
            print(f"start epoch: {start_epoch}")
            print(f"start acc: {start_acc}")

    trainer = Trainer(args, train_dataset)


    for epoch in range(start_epoch, start_epoch + args.epochs):
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

    start_acc = test_acc if not args.cifar100 else test_acc1
    state = {
        'client': client_part.cpu().state_dict(),
        'server': server_part.cpu().state_dict(),
        'epoch': str(start_epoch + args.epochs),
        'acc': str(start_acc)
    }
    torch.save(state,  '../checkpoints/' + args.reload_path + '_ckpt')
    print("==> saving checkpoint....")
    
    
