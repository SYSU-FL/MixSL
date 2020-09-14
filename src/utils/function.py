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


def weight_assign(args):
    m = max(int(args.num_users * args.frac), 1)
    if m % 2 != 0:
        a, b =  int(m//2), int(m//2 + 1)
    else:
        a, b = int(m/2), int(m/2)
    idxs_users0 = np.random.choice(range(80), a, replace=False)
    idxs_users1 = np.random.choice(range(80,100), b, replace=False)
    idxs_users = np.append(idxs_users0, idxs_users1)
    print(f"idxs_users: {idxs_users}")
    return idxs_users



def pre_train(args, client, server, dataset):
    print("==========pre_train using the mnist=======================")
    #conv = nn.Conv2d(1, 3, kernel_size=1)
    client.train()
    server.train()
    criterion = nn.CrossEntropyLoss()
    optimizer0 = torch.optim.SGD(client.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    optimizer1 = torch.optim.SGD(server.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
    if args.gpu > -1:
        #conv.cuda(args.gpu)
        criterion.cuda(args.gpu)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for epoch in range(5):
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"pre train epoch: {epoch}, batchid: {batch_idx}")
            images, labels = images.to(args.gpu), labels.to(args.gpu)
            # with torch.no_grad():
            #     images = conv(images)
            #torch.cuda.empty_cache()
            client.zero_grad()
            server.zero_grad()
            activations = client(images)
            a_temp = activations.clone().detach()
            a_temp.requires_grad_(True)
            outputs = server(a_temp)
            loss = criterion(outputs, labels)
            loss.backward()
            grad = copy.deepcopy(a_temp.grad.data)
            activations.backward(grad)
            optimizer1.step()
            optimizer0.step()
    print("==============pre_train end......")
    return client.state_dict(), server.state_dict()



# def noisy(sensitivety, epsilon):
#     beta = sensitivety / epsilon
#     u1 = np.random.random()
#     u2 = np.random.random()
#     if u1 <= 0.5:
#         n_value = -beta * np.log(1.-u2)
#     else:
#         n_value = beta * np.log(u2)
#     return n_value


# def differential_privacy(data, sensitivety=1, epsilon=1):
#     size = data.size()
#     data = data.view(-1)
#     for i in range(len(data)):
#         data[i] += noisy(sensitivety, epsilon)
#     data = data.view(size)
#     return data

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

    
if __name__ == "__main__":
    # m = max(int(50 *1), 1)
    # idxs_users = np.arange(m)
    # a = [i for i in idxs_users]
    # print(a)
    before = torch.randn(2,3,4)
    print(before)
    print("----------")
    print(differential_privacy(before))