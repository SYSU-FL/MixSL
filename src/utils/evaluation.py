import os
import pandas as pd
from datetime import datetime
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import copy



"""
for single training on cifar10
"""

def test_inference(args, net, test_dataset):
    """ Returns the test accuracy and loss.
    """
    net.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    accuracy = correct/total
    return accuracy, loss

"""
for single training in cifar100
"""
def test_inference4cifar100(args, net, test_dataset):
    """ Returns the test accuracy and loss.
    """
    net.eval()
    loss, total, correct_1, correct_5 = 0.0, 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1 
            correct_1 += correct[:, :1].sum()

    accuracy_1 = correct_1 / len(testloader.dataset)
    accuracy_5 = correct_5 / len(testloader.dataset)
    return accuracy_1, accuracy_5, loss



def test_inference4split(args, client, server, test_dataset):
    """ Returns the test accuracy and loss.
    """
    client.eval()
    server.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = client(images)
            outputs = server(outputs)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    accuracy = correct/total
    return accuracy, loss


def test_inference4split4cifar100(args, client, server, test_dataset):
    """ Returns the test accuracy and loss.
    """
    client.eval()
    server.eval()
    loss, total, correct, correct_1, correct_5 = 0.0, 0.0, 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()
    if args.gpu > -1:
        criterion.cuda(args.gpu)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            if args.gpu > -1:
                images, labels = images.cuda(args.gpu), labels.cuda(args.gpu)

            # Inference
            outputs = client(images)
            outputs = server(outputs)
            batch_loss = criterion(outputs, labels)
            loss += copy.deepcopy(batch_loss.item())

           # Prediction
            _, pred = outputs.topk(5, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1 
            correct_1 += correct[:, :1].sum()

    accuracy_1 = correct_1 / len(testloader.dataset)
    accuracy_5 = correct_5 / len(testloader.dataset)
    return accuracy_1, accuracy_5, loss





def save_measure(logs, model_name):
    if not os.path.exists("../measure"):
        os.makedirs("../measure")
    df = pd.DataFrame(logs)
    path = '../measure/weight-diff-{}-{}'.format(model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a')

        
