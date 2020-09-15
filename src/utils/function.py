import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision
from scipy.spatial.distance import pdist, squareform

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
