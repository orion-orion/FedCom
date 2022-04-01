# -*- coding: utf-8 -*-
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-12-24 14:33:14
LastEditors: ZhangHongYu
LastEditTime: 2022-04-01 20:01:36
'''
import torch
import pandas as pd
import numpy as np
from utils.plots import ExperimentLogger
from method.clustered.clustered import clustered_fl
from method.my.my import my_fl
from method.fedavg.fedavg import fed_avg_fl
from method.ditto.ditto import ditto_fl
from method.local.local import local_fl
from init_devices import init_clients_and_server
from init_datasets import load_dataset
import argparse
import os


def parse_args():
    """parse the command line args

    Returns:
        args: a namespace object including args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'dataset',
        help="name of dataset;"
        " possible are `EMNIST`, `FashionMNIST`, `CIFAR10`，`CIFAR100`, `Shakespeare`",
        type=str,
        default='CIFAR10'
    )
    parser.add_argument(
        'method',
        help = "the method to be used;"
               " possible are `My`,`Clustered`, `FedAvg`, `Ditto`, `Local`",
        type=str,
        default='My'
    )
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_clients',
        help = "the number of clients",
        type=int,
        default=10
    )
    parser.add_argument(
        '--n_clusters',
        help = "initialize the number of cluster of data distribution",
        type=int,
        default=3
    )
    parser.add_argument(
        '--alpha',
        help = "the parameter of dirichlet",
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--local_epochs',
        help='number of local epochs before communication; default is 1',
        type=int,
        default=1
    )
    parser.add_argument(
        "--n_rounds",
        help="number of communication rounds",
        type=int,
        default=100
    )
    parser.add_argument(
        "--n_sample",
        help="number of sample to use",
        type=int,
        default=20000
    )
    parser.add_argument(
        "--train_frac",
        help="fraction of train samples",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--val_frac",
        help="fraction of validation samples in train samples",
        type=float,
        default=0
    )   
    parser.add_argument(
        "--seed",
        help='random seed',
        type=int,
        default=42
    )
    args = parser.parse_args()
    return args

def run_experiment(args, clients, server):

                
    cfl_stats = ExperimentLogger()

    if args.method == "Clustered":
        clustered_fl(args, clients, server, cfl_stats)
    elif args.method == "My":
        my_fl(args, clients, server, cfl_stats)    
    elif args.method == "FedAvg":
        fed_avg_fl(args, clients, server, cfl_stats)
    elif args.method == "Ditto":
        ditto_fl(args, clients, server, cfl_stats)
    elif args.method == "Local":
        local_fl(args, clients, server, cfl_stats)
    else:
        raise IOError("possible are `My`,`Clustered`, `FedAvg`, `Ditto`, `Local`")

    # The training process resulted in multiple models for every client: A Federated Learning base model 
    # as well as more specialized models for the different clusters.  We can now compare their accuracies
    # on the clients' validation sets, and assign each client the model which performed best.
    results = np.zeros([len(clients), len(server.model_cache)])
    for i, (idcs, W, accs) in enumerate(server.model_cache):
        results[idcs, i] = np.array(accs)

    frame = pd.DataFrame(results, columns=["FL Model"]+["Model {}".format(i) 
                                                        for i in range(results.shape[1]-1)],
                index = ["Client {}".format(i) for i in range(results.shape[0])])

    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]
    frame = frame.T.style.apply(highlight_max)
    path = "result_pic"
    if not os.path.exists(path):
        os.makedirs(path)
    frame.to_html(os.path.join(path, "specialized_acc.html"))
    # As we can see, clustering improoved the accuracy for all clients by about 10%.

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    dataset, client_train_idcs, client_test_idcs, client_val_idcs, data_info = load_dataset(args)
    # print(dataset[0])
    # print(client_train_idcs)
    # print(client_test_idcs)
    clients, server = init_clients_and_server(args, dataset, client_train_idcs, client_test_idcs, client_val_idcs, data_info)

    #Now everything is set up to run our Clustered Federated Learning algorithm. During training, we will track the mean and std client accuracies, as well as the average and maximum client update norms.
 
    run_experiment(args, clients, server)

if __name__ == "__main__":
    main()

    
    



