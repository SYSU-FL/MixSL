import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
import numpy as np
from .imbalance_cifar import IMBALANCECIFAR10, RemainData, IMBALANCECIFAR100

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(int(label))#  根据imbalanced-cifar10修改了这里

def get_cifar10(balanced=False, remain_flag=False):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if balanced:
        trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset
    else:
        trainset = IMBALANCECIFAR10(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
        if remain_flag:
            remain_data, remain_labels = trainset.get_remain_data()
            remain_dataset = RemainData(root='../data', remain_data=remain_data, remain_labels=remain_labels, train=True, download=True, transform=transform_train)
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}, remain dat size: {len(remain_dataset)}")
            return trainset, testset, remain_dataset
        else:
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
            return trainset, testset



def get_cifar100(balanced=False, remain_flag=False):
    transform_train = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        np.array([125.3, 123.0, 113.9]) / 255.0,
        np.array([63.0, 62.1, 66.7]) / 255.0),])

    if balanced:
        trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
        return trainset, testset
    else:
        trainset = IMBALANCECIFAR100(root='../data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
        if remain_flag:
            remain_data, remain_labels = trainset.get_remain_data()
            remain_dataset = RemainData(root='../data', remain_data=remain_data, remain_labels=remain_labels, train=True, download=True, transform=transform_train)
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}, remain dat size: {len(remain_dataset)}")
            return trainset, testset, remain_dataset
        else:
            print(f"train data size: {len(trainset)}, test data size: {len(testset)}")
            return trainset, testset

def random_avg_strategy(trainset, num=100):
    num_items = int(len(trainset)/num)
    dict_users, all_idxs = {}, [i for i in range(len(trainset))]
    for i in range(num):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users





def count_class_num_per_client(train_set, groups, num_classes=10):
    g2c = {}
    for i, elems in groups.items():
        count = {i: 0 for i in range(0,num_classes)}
        for e in elems:
            img, label = train_set[e]
            count[label] += 1
        vals = count.values()
        g2c[i] = vals
    return g2c

def count_class_num(train_set, num_classes=10):
    count = {i: 0 for i in range(0, num_classes)}
    size = len(train_set)
    for i in range(size):
        _, label = train_set[i]
        count[label] += 1
    return count

