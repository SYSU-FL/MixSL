import pandas as pd
from datetime import datetime
import time
import numpy as np
import random
import torch

def format_args(args):
    return "frac{}-bs{}-users{}-epochs{}-k(onlyformixup){}-pai(onlyformixup){}-m{}-lr{}".format(args.frac, args.local_bs, args.num_users, args.epochs, args.k, args.pai, args.momentum, args.lr)

def save_logs(logs, tag, args):
    df = pd.DataFrame(logs)
    param_str = format_args(args)
    path = '../logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')
    df['test_acc'] = df['test_acc'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy: {df.loc[:,'test_acc'].max()}")
    print("save logs success!")

def save_cifar100_logs(logs, tag, args):
    df = pd.DataFrame(logs)
    param_str = format_args(args)
    path = '../logs/{}_{}_{}.csv'.format(tag, param_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.to_csv(path, mode='a',index_label='index')

    df['test_acc1'] = df['test_acc1'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy1: {df.loc[:,'test_acc1'].max()}")
    df['test_acc5'] = df['test_acc5'].apply(lambda x: float(x.replace('%', '')))
    print(f"final Accuracy5: {df.loc[:,'test_acc5'].max()}")
    print("save logs sucess!")

def setup_seed(seed, gpu_enabled):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if gpu_enabled:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def write_file(name, *lines):
    with open('../measure/' + name + '.txt', 'w') as f:
        for l in lines:
            line = [str(i) for i in l]
            f.write(' '.join(line) + '\n')


        
