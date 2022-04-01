'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-13 21:16:56
LastEditors: ZhangHongYu
LastEditTime: 2022-04-01 19:54:13
'''
from models import ConvNet, MobileNet, NextCharacterLSTM
from utils.data_utils import rotate_data
from custom_ds.subsets import CustomSubset
from utils.plots import display_sample
from fl_devices import Client, Server
from method.ditto.client import DittoClient
from method.clustered.server import ClusteredServer
from method.my.server import CommunityServer

import torch

def init_clients_and_server(args, dataset, client_train_idcs, client_test_idcs, client_val_idcs, data_info):

    # 获取训练集相关属性
    if args.dataset != "Shakespeare":
        n_channels, classes, input_sz, num_cls =\
            data_info['n_channels'], data_info['classes'], data_info['input_sz'], data_info['num_cls']

    client_train_data = [CustomSubset(dataset, idcs) for idcs in client_train_idcs]
    client_test_data = [CustomSubset(dataset, idcs) for idcs in client_test_idcs]
    client_val_data = [CustomSubset(dataset, idcs) for idcs in client_val_idcs]


  

    # 对所有数据旋转以模拟簇状结构。即先iid再划分簇的方式，同一个client保持旋转模式一致
    # 虽然对簇划分算法能抵御这种扰动，但会降低传统联邦学习算法精度，有点可以为之的感觉。
    if args.dataset != 'Shakespeare':
        rotate_data(client_train_data, client_test_data, client_val_data, args.n_clients, args.n_clusters)     
    
    if args.dataset != 'Shakespeare':
        if args.method == "Ditto":
            clients = [DittoClient(lambda: ConvNet(input_size=input_sz, channels=n_channels, num_classes=num_cls),\
                lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9), train_data = train_dat, test_data = test_dat, val_data = val_dat, idnum=i) \
                    for i, (train_dat, test_dat, val_dat) in enumerate(zip(client_train_data, client_test_data, client_val_data))]
        else:
            clients = [Client(lambda: ConvNet(input_size=input_sz, channels=n_channels, num_classes=num_cls),\
            lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9), train_data = train_dat, test_data = test_dat, val_data = val_dat, idnum=i) \
                for i, (train_dat, test_dat, val_dat) in enumerate(zip(client_train_data, client_test_data, client_val_data))]
                # clients = [Client(lambda: MobileNet(num_classes=num_cls), lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9), \
        #     train_data = train_dat, test_data = test_dat, val_dat = val_dat, idnum=i) \
        #         for i, (train_dat, test_dat, val_dat) in enumerate(zip(client_train_data, client_test_data, client_val_data))]

                
        if args.method == "Clustered":
            server = ClusteredServer(lambda : ConvNet(input_size=input_sz, channels=n_channels, num_classes=num_cls))
        elif args.method == "My":
            server = CommunityServer(lambda : ConvNet(input_size=input_sz, channels=n_channels, num_classes=num_cls))
        else:
            server = Server(lambda : ConvNet(input_size=input_sz, channels=n_channels, num_classes=num_cls))

        #将optim优化函数封装为一个匿名函数，然后其它参数就可以提供默认值
         # server = Server(lambda : MobileNet(num_classes=num_cls))
    else:
        if args.method == "Ditto":
            clients = [DittoClient(lambda: NextCharacterLSTM(), lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9),\
                train_data = train_dat, test_data = test_dat, val_data = val_dat, idnum=i) for i, (train_dat, test_dat, val_dat) \
                    in enumerate(zip(client_train_data, client_test_data, client_val_data)) if len(train_dat)> 0 ]
        else:
            clients = [Client(lambda: NextCharacterLSTM(), lambda x : torch.optim.SGD(x, lr=0.1, momentum=0.9),\
            train_data = train_dat, test_data = test_dat, val_data = val_dat, idnum=i) for i, (train_dat, test_dat, val_dat) \
                in enumerate(zip(client_train_data, client_test_data, client_val_data)) if len(train_dat)> 0 ]


        if args.method == "Clustered":
            server = ClusteredServer(lambda : NextCharacterLSTM())
        elif args.method == "My":
            server = CommunityServer(lambda : NextCharacterLSTM())
        else:
            server = Server(lambda : NextCharacterLSTM())


    # 初始化各client的权重
    client_n_samples = torch.tensor([client.n_train_samples for client in clients])
    samples_sum = client_n_samples.sum()
    for client in clients:
        client.weight = client.n_train_samples/samples_sum


    # 从现有client中抽取部分样本进行可视化
    # if args.dataset != "Shakespeare":
    #     display_sample(clients, classes)
    return clients, server
