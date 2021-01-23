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
from data_utils import get_cifar10, DatasetSplit, sequence_avg_strategy, random_avg_strategy, get_cifar100
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
    
def client_training(model, loader, s, args):
    start = time.time()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        iter_start = time.time()
        model.zero_grad()
        images = images.cuda(args.local_rank)
        outputs = model(images)
        if args.dp:
            outputs = differential_privacy(outputs, args.gpu)
        client_outputs = outputs.clone().detach().cpu().numpy()
        msg = {
            'client_output': client_outputs,
            'label': labels.numpy()
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
    criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    model.train()
    for e in range(args.epochs):
        for i in range(total_batch):
            optimizer.zero_grad()
            msg, datasize = recv_msg(conn)
            recv_datasize_list.append(datasize)
            msg_outputs = msg['client_output']
            msg_label = msg['label']
            client_outputs = torch.from_numpy(msg_outputs).cuda(args.local_rank)
            client_label = torch.from_numpy(msg_label).cuda(args.local_rank)
            client_outputs.requires_grad_(True)
            outputs = model(client_outputs)
            loss = criterion(outputs, client_label)
            loss.backward()
            msg = copy.deepcopy(client_outputs.grad.data)
            msg = msg.cpu().numpy()
            datasize = send_msg(conn, msg)
            send_datasize_list.append(datasize)
            optimizer.step()
    end = time.time()
    return end - start
    





    

if __name__ == "__main__":
    args = parse_args()
    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    TAG = 'sl-ditributed' + data_name + '-' + args.name
    print(f'{TAG}: training start....')
    setup_seed(args.seed, True if args.gpu > -1 else False)
    
    logs = []
    if args.cifar100:
        train_dataset, test_dataset = get_cifar100(args.balanced)
    else:
        train_dataset, test_dataset = get_cifar10(args.balanced)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
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
        training_time = client_training(model, trainloader, s, args)
        print(f"training time: {training_time}")
        write_cost(TAG + '--senddata', send_datasize_list)
        write_cost(TAG + '--recvdata', recv_datasize_list)
        write_cost(TAG + '--transfer', transfer_time_list)
        write_cost(TAG + '--computer', compute_time_list)
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



   
    