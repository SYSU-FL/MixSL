import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.remain_data = []
        self.remain_labels = []
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        if not hasattr(self, 'data'):
            if self.train:
                img_max = len(self.train_data) / cls_num # in high version train_data is data
            else:
                img_max = len(self.test_data) / cls_num # in high version train_data is data
        else:
            img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        print(img_num_per_cls)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        if not hasattr(self, 'targets'):
            if self.train:
                targets_np = np.array(self.train_labels, dtype=np.int64) #in high version, self.train_labels is self.targets
            else:
                targets_np = np.array(self.test_labels, dtype=np.int64) #in high version, self.train_labels is self.targets
        else:
            targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            remain_idx = idx[the_img_num: ]
            if self.train:
                new_data.append(self.train_data[selec_idx, ...] if not hasattr(self, 'data') else self.data[selec_idx, ...])
                remain_size = len(self.train_data[remain_idx, ...] if not hasattr(self, 'data') else self.data[remain_idx, ...])
                self.remain_data.append(self.train_data[remain_idx, ...] if not hasattr(self, 'data') else self.data[remain_idx, ...])
                self.remain_labels.extend([the_class,] * remain_size)
            else:
                new_data.append(self.test_data[selec_idx, ...] if not hasattr(self, 'data') else self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
            
            
        new_data = np.vstack(new_data)
        if self.train:
            self.remain_data = np.vstack(self.remain_data)
            if not hasattr(self, 'data'):
                self.train_data = new_data
                self.train_labels = new_targets
            else:
                self.data = new_data
                self.targets = new_targets
        else:
            if not hasattr(self, 'data'):
                self.test_data = new_data
                self.test_labels = new_targets
            else:
                self.data = new_data
                self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def get_remain_data(self):
        return self.remain_data, self.remain_labels

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100

class RemainData(torchvision.datasets.CIFAR10):
    def __init__(self, root, remain_data, remain_labels, train=True, download=True, transform=None, target_transform=None):
        super(RemainData, self).__init__(root, train, transform, target_transform, download)
        if not hasattr(self, 'data'):
            self.train_data = remain_data
            self.train_labels = remain_labels
        else:
            self.data = remain_data
            self.targets = remain_labels

        


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()