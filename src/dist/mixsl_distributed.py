import os
import copy
import time
import pickle
import struct
import socket
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
from split_models import CNN_Extractor, CNN_Classifer
from options import parse_args
from data_utils import get_cifar10, DatasetSplit, sequence_avg_strategy, random_avg_strategy, get_cifar100, count_class_num
from utils import save_logs, save_cifar100_logs, setup_seed, write_file, write_cost
from function import random_assign, weight_assign, naive_assign, pre_train, differential_privacy, distcorr, compute_dcor
from resnet import ResNet18_Extractor, ResNet18_Classifer, ResNet34_Extractor, ResNet34_Classifer
from vgg import get_split_vgg16
from evaluation import test_inference4split, test_inference4split4cifar100
from communication import *
#from tqdm import tqdm





send_datasize_list = []
recv_datasize_list = []
transfer_time_list = []
compute_time_list = []




def max_labels(y_a, y_b, lam_ys):
    max_labs = []
    for i in range(len(lam_ys)):
        max_labs.append(y_a[i] if lam_ys[i] > 0.5 else y_b[i])
    return torch.Tensor(max_labs).type_as(y_a).view(y_a.size()).cuda(args.gpu)
    
    


def get_labels_factors(a, b, lam, num_per_cls,  k=2, pai=0.5):
    lam_y = 0.
    if (num_per_cls[a.item()] / num_per_cls[b.item()] >= k) and (lam < pai):
        lam_y = 0.
    elif (num_per_cls[a.item()] / num_per_cls[b.item()] <= 1) and ((1 - lam) < pai):
        lam_y = 1.
    else:
        lam_y = lam
    return lam_y

def mixup_data(x, y,c2n):
    lam = np.random.beta(1., 1.)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    lam_ys = []
    for i in range(y_a.size(0)):
        lam_ys.append(get_labels_factors(y_a[i], y_b[i], lam, c2n))
    return mixed_x, y_a, y_b, lam, lam_ys

def mixup(x, y, c2n, num=3):
    x_ = x
    y_ = y
    lam_yss = []
    label_tuples = []
    for i in range(num-1):
        mixed_x, y_a, y_b, lam, lam_ys = mixup_data(x_, y_, c2n)
        lam_yss.append(lam_ys)
        if i == 0:
            label_tuples.append(y_a)
        label_tuples.append(y_b)
        x_ = mixed_x
        y_ = max_labels(y_a, y_b, lam_ys)
    return mixed_x, label_tuples, lam_yss


    
def client_training(model, loader, s, args, c2n):
    start = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        iter_start = time.time()
        model.zero_grad()
        images = images.cuda(args.local_rank)
        outputs = model(images)
        outputs, labels_vectors, lam_yss = mixup(outputs, labels, c2n, args.mix_num)
        client_outputs = outputs.clone().detach().cpu().numpy()
        label_numpys = [label.cpu().numpy() for label in labels_vectors]
        msg = {
            'client_output': client_outputs,
            'label': label_numpys,
            'lamy': lam_yss
        }
        send_start = time.time()
        datasize = send_msg(s, msg)
        send_end = time.time()
        send_datasize_list.append(datasize)
        recv_start = time.time()
        grad, datasize = recv_msg(s)
        recv_end = time.time()
        recv_datasize_list.append(datasize)
        grad = torch.from_numpy(grad).cuda(args.local_rank)
        outputs.backward(grad)
        optimizer.step()
        iter_end = time.time()
        transfer_time = (send_end - send_start) + (recv_end - recv_start) 
        transfer_time_list.append(transfer_time)
        compute_time = (iter_end - iter_start) - transfer_time
        compute_time_list.append(compute_time)
    end = time.time()
    return end - start
        

def server_training(model, conn, total_batch, args):
    start = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    #criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    model.train()
    client_label_list = []
    for e in range(args.epochs):
        for i in range(total_batch):
            optimizer.zero_grad()
            msg, datasize = recv_msg(conn)
            recv_datasize_list.append(datasize)
            msg_outputs = msg['client_output']
            msg_label_list = msg['label']
            lamy_list = msg['lamy']
            client_outputs = torch.from_numpy(msg_outputs).cuda(args.local_rank)
            client_label_list = [torch.from_numpy(label).cuda(args.local_rank) for label in msg_label_list]
            client_outputs.requires_grad_(True)
            outputs = model(client_outputs)
            loss = multloss(outputs, client_label_list, lamy_list)
            loss.backward()
            msg = copy.deepcopy(client_outputs.grad.data)
            msg = msg.cpu().numpy()
            datasize = send_msg(conn, msg)
            send_datasize_list.append(datasize)
            optimizer.step()
    end = time.time()
    return end - start
    
def multloss(outputs, labels_vectors, lam_yss):
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    lam_y = torch.FloatTensor(lam_yss[0]).cuda(args.gpu)
    one_vector = torch.ones(lam_y.size()).cuda(args.gpu)
    loss = (lam_y * criterion(outputs, labels_vectors[0])) + ((one_vector - lam_y) * criterion(outputs, labels_vectors[1]))
    for i in range(1, len(lam_yss)):
        lam_y = torch.FloatTensor(lam_yss[i]).cuda(args.gpu)
        one_vector = torch.ones(lam_y.size()).cuda(args.gpu)
        loss = (lam_y * loss) + ((one_vector - lam_y) * criterion(outputs, labels_vectors[i + 1]))
    loss = loss.mean()
    return loss




    

if __name__ == "__main__":
    args = parse_args()
    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    TAG = 'mixsl-ditributed' + data_name + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    
    logs = []
    if args.cifar100:
        train_dataset, test_dataset = get_cifar100(args.balanced)
    else:
        train_dataset, test_dataset = get_cifar10(args.balanced)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    c2n = count_class_num(train_dataset, 100)
    logs_file = TAG
    flag = True
    #get_split_vgg16(num_classes)
    if args.local_rank == 0:
        model = ResNet18_Extractor().cuda(args.local_rank)
        #model = ResNet34_Extractor().cuda(args.local_rank)
    elif args.local_rank == 1:
        model = ResNet18_Classifer(num_classes).cuda(args.local_rank)
        #model = ResNet34_Classifer(num_classes).cuda(args.local_rank)

    if args.local_rank == 0:
        s = socket.socket()
        s.connect((args.ip, args.port))
        total_batch = len(trainloader)
        send_msg(s, total_batch)
        training_time = client_training(model, trainloader, s, args, c2n)
        print(f"training time: {training_time}")
        write_cost(TAG + str(args.mix_num) + '--senddata', send_datasize_list)
        write_cost(TAG + str(args.mix_num) +'--recvdata', recv_datasize_list)
        write_cost(TAG + str(args.mix_num) +'--transfer', transfer_time_list)
        write_cost(TAG + str(args.mix_num) +'--computer', compute_time_list)
        s.close()
    elif args.local_rank == 1:
        s = socket.socket()
        s.bind((args.ip, args.port))
        s.listen(5)
        conn, addr = s.accept()
        total_batch, datasize = recv_msg(conn)
        training_time = server_training(model, conn, total_batch, args)
        print(f"training time: {training_time}")
        conn.close()
        s.close()



   
    