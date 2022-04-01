'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-13 21:24:29
LastEditors: ZhangHongYu
LastEditTime: 2022-03-29 16:46:00
'''

from torch.utils.data import  ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
from custom_ds.femnist import FEMNIST
from custom_ds.shakespeare import Shakespeare
from utils.plots import display_data_distribution
from utils.data_utils import split_dataset_by_mixture_distribution, split_noniid, pathological_non_iid_split
from custom_ds.subsets import CustomSubset
import os
import numpy as np

def load_dataset(args):
    """load train and test dataset

    Args:
        args: the namespace object including args

    Returns:
        train_data: train_data
    """
    if not os.path.exists("./data"):
        os.mkdir("./data")

    data_info = {}
    if args.dataset == "Shakespeare":
        dataset = Shakespeare(root="./data", download=True, train_frac = args.train_frac, val_frac=args.val_frac)
        # Shakespeare数据集的样本个数由n_client决定
        client_train_idcs, client_test_idcs = \
            dataset.client_train_idcs[:args.n_clients], dataset.client_test_idcs[:args.n_clients]
        if args.val_frac > 0:
            client_val_idcs = dataset.client_val_idcs[:args.n_clients]  
        else:
            client_val_idcs = [[] for i in range(args.n_clients)] 
    elif args.dataset == "FEMNIST":
        transform = transforms.Compose(
            [
                ToTensor(),
            ]
        )
        dataset = FEMNIST(root="./data", download=True, train_frac = args.train_frac, val_frac=args.val_frac,transform=transform)
        # FEMNIST数据集的样本个数由n_client决定
        client_train_idcs, client_test_idcs = \
            dataset.client_train_idcs[:args.n_clients], dataset.client_test_idcs[:args.n_clients]
        if args.val_frac > 0:
            client_val_idcs = dataset.client_val_idcs[:args.n_clients]  
        else:
            client_val_idcs = [[] for i in range(args.n_clients)] 
        if len(dataset.data[0].shape) == 2:
            data_info['n_channels'] = 1
        else:
            data_info['n_channels'] = dataset.data[0].shape[2]
        data_info['classes'] = dataset.n_classes
        data_info['input_sz'], data_info['num_cls'] = dataset.data[0].shape[0],  dataset.n_classes
    else:
        if args.dataset == "EMNIST":
            transform = transforms.Compose(
                [
                    ToTensor(),
                ]
            )
                # train = True，从训练集create数据
            train_data = datasets.EMNIST(root="./data", split="byclass", download=True, transform=transform, train=True)
            # test = False，从测试集create数据
            test_data = datasets.EMNIST(root="./data", split="byclass", download=True, transform=transform, train=False)
        elif args.dataset == "FashionMNIST":
            transform = transforms.Compose(
                [
                    ToTensor(),
                ]
            )
            # train = True，从训练集create数据
            train_data = datasets.FashionMNIST(root="./data", download=True, transform=transform, train=True)
            # test = False，从测试集create数据
            test_data = datasets.FashionMNIST(root="./data", download=True, transform=transform, train=False)
        elif args.dataset == "CIFAR10":
            transform = transforms.Compose(
                [
                    ToTensor(),
                    # Normalize(
                    #     (0.4914, 0.4822, 0.4465),
                    #     (0.2023, 0.1994, 0.2010)
                    # ),
                    # ToPILImage()
                    # transforms.Grayscale(num_output_channels=1)
                ]
            )
            train_data = datasets.CIFAR10(root="./data", download=True, transform=transform, train=True)
                # test = False，从测试集create数据
            test_data = datasets.CIFAR10(root="./data", download=True, transform=transform, train=False)
        elif args.dataset == "CIFAR100":
            transform = transforms.Compose(
                [
                    # transforms.Grayscale(num_output_channels=1)
                    ToTensor(),
                ]
            )
                # train = True，从训练集create数据
            train_data = datasets.CIFAR100(root="./data", download=True, transform=transform, train=True)
                # test = False，从测试集create数据
            test_data = datasets.CIFAR100(root="./data", download=True, transform=transform, train=False)
        else:
            raise IOError("Please input the correct dataset name, it must be one of:"
                        "EMNIST, FashionMNST, CIFAR10, CIFAR100 and Shakespeare.")
                        
        # 获取训练集相关属性
        if len(train_data.data[0].shape) == 2:
            data_info['n_channels'] = 1
        else:
            data_info['n_channels'] = train_data.data[0].shape[2]
        data_info['classes'] = train_data.classes
        data_info['input_sz'], data_info['num_cls'] = train_data.data[0].shape[0],  len(train_data.classes)
        labels = np.concatenate([np.array(train_data.targets), np.array(test_data.targets)], axis=0)
        dataset= ConcatDataset([train_data, test_data]) 



        # 按混合分布划分，增强节点之间数据相似性
        # client_idcs = split_dataset_by_mixture_distribution(CustomSubset(dataset, np.arange(args.n_sample)), \
        #     n_classes = data_info['num_cls'], n_clients=args.n_clients, n_clusters=3, alpha=args.alpha)
        
        if args.pathological_split:
            # 每个client默认两种label的样本
            client_idcs = pathological_non_iid_split(CustomSubset(dataset, np.arange(args.n_sample)), \
                n_classes = data_info['num_cls'], n_clients=args.n_clients, n_classes_per_client=args.n_shards)
        else:
            # 注意每个client不同label的样本数量不同，以此做到non-iid划分, 数据集只用前n_sample个样本
            client_idcs = split_noniid(labels[:args.n_sample], alpha=args.alpha, n_clients=args.n_clients)
            

        #display_data_distribution(client_idcs, labels, data_info['num_cls'], args.n_clients, data_info['classes'])


        client_train_idcs, client_test_idcs, client_val_idcs = [], [], []
        for idcs in client_idcs:
            n_samples = len(idcs)
            n_train = int(n_samples * args.train_frac)
            n_test = n_samples - n_train
            if args.val_frac > 0:
                n_val = int(n_train * (1-args.val_frac))
                n_train = n_train * args.val_frac
                client_val_idcs.append(idcs[n_train:(n_train+n_val)])
            else:
                client_val_idcs.append([])
            client_train_idcs.append(idcs[:n_train])
            client_test_idcs.append(idcs[n_test:])
    
    return dataset, client_train_idcs, client_test_idcs, client_val_idcs, data_info

