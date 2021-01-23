import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from scipy.spatial.distance import pdist, squareform
from ..model.resnet.ResNet18_Extractor, ResNet18_Classifer,ResNet34_Extractor,ResNet34_Classifer
from ..model.vgg.get_split_vgg16
from ..dataprocess.data_utils from get_cifar100, get_cfiar10, cifar_noniid, \
    random_avg_strategy,fix_count_class_num_per_client,count_class_num_per_client


def assgin_task(args):
    data_name = 'cifar10' if not args.cifar100 else 'cifar100'
    num_classes = 10 if not args.cifar100 else 100
    split_type = 'googlesplit' if args.google_split else ''
    if not args.google_split:
        if args.cifar100:
            train_dataset, test_dataset = get_cifar100(args.balanced)
        else:
            train_dataset, test_dataset = get_cifar10(args.balanced)
        user_groups = random_avg_strategy(train_dataset, args.num_users)
        cls_num_per_clients = count_class_num_per_client(train_dataset, user_groups, 100)
    else:
        if args.cifar100:
            train_dataset, test_dataset = get_cifar100(True)
        else:
            train_dataset, test_dataset = get_cifar10(True)
        user_groups, ps = cifar_noniid(train_dataset, args.num_users, True)
        user_groups = fix_count_class_num_per_client(user_groups)
        cls_num_per_clients =  user_groups
    if args.model == 'VGG16':
        client_part, server_part = get_split_vgg16(num_classes)
    else:
        client_part = eval(args.model + '_Extractor()')
        server_part= eval(args.model + '_Classifer(num_classes)') 
    if args.gpu > -1:
        client_part.cuda(args.gpu)
        server_part.cuda(args.gpu)
    return client_part, server_part,\ 
            train_dataset, test_dataset,\
            user_groups , cls_num_per_clients,\
            data_name "-" + split_type



def random_assign(args):
    m = max(int(args.num_users * args.frac), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    return idxs_users






def differential_privacy(data, cuda=0):
    noise = torch.FloatTensor(data.shape).normal_(0, 1.1)
    if cuda > -1:
        noise = noise.to(cuda)
    data.add_(noise)
    return data

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

    
def compute_dcor(x, y, args):

    def _distance_covariance(a_matrix, b_matrix):
        return (a_matrix * b_matrix).sum().sqrt() / a_matrix.size(0)

    def _distance_variance(a_matrix):
        return (a_matrix ** 2).sum().sqrt() / a_matrix.size(0)

    def _A_matrix(data):
        distance_matrix = _distance_matrix(data)

        row_mean = distance_matrix.mean(dim=0, keepdim=True)
        col_mean = distance_matrix.mean(dim=1, keepdim=True)
        data_mean = distance_matrix.mean()

        return distance_matrix - row_mean - col_mean + data_mean

    def _distance_matrix(data):
        n = data.size(0)
        distance_matrix = torch.zeros((n, n)).cuda(args.gpu)

        for i in range(n):
            for j in range(n):
                row_diff = data[i] - data[j]
                distance_matrix[i, j] = (row_diff ** 2).sum()

        return distance_matrix

    input_data = x.clone().detach()
    intermediate_data = y.clone().detach()
    input_data = input_data.view(input_data.size(0), -1)
    intermediate_data = intermediate_data.view(intermediate_data.size(0), -1)

    # Get A matrices of data
    A_input = _A_matrix(input_data)
    A_intermediate = _A_matrix(intermediate_data)

    # Get distance variances
    input_dvar = _distance_variance(A_input)
    intermediate_dvar = _distance_variance(A_intermediate)

    # Get distance covariance
    dcov = _distance_covariance(A_input, A_intermediate)

    # Put it together
    dcorr = dcov / (input_dvar * intermediate_dvar).sqrt()

    return dcorr
