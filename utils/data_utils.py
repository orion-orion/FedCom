'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-12-23 18:53:12
LastEditors: ZhangHongYu
LastEditTime: 2022-03-26 19:37:23
'''
import numpy as np
from torchvision import transforms

def split_noniid(train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    Args:
        train_labels: ndarray of train_labels
        alpha: the parameter of dirichlet distribution
        n_clients: number of clients
    Returns:
        client_idcs: a list containing sample idcs of clients
    """

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (62, 10) 记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # (62, ...)，记录每个类别对应的样本下标

    
    client_idcs = [[] for _ in range(n_clients)]
    # 记录每个client对应的样本下标
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将样本划分为了对应的份数
        # for i, idcs 为遍历每一个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs


def split_dataset_by_mixture_distribution(dataset, n_classes, n_clients, n_clusters, alpha):
    if n_clusters == -1:
        n_clusters = n_classes
        
    all_labels = list(range(n_classes))
    np.random.shuffle(all_labels)
    
    def iid_divide(l, g):
        """
        将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
        每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
        返回由不同的groups组成的列表
        """
        num_elems = len(l)
        group_size = int(len(l) / g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(l[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
        return glist
    clusters_labels = iid_divide(all_labels, n_clusters)

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx

    data_idcs = list(range(len(dataset)))
    
    
    clusters_sizes = np.zeros(n_clusters, dtype=int)
    # 存储每个cluster对应的数据索引
    clusters = {k: [] for k in range(n_clusters)}
    for idx in data_idcs:
        _, label = dataset[idx]
        # 由样本数据的label先找到其cluster的id
        group_id = label2cluster[label]
        # 再将对应cluster的大小+1
        clusters_sizes[group_id] += 1
        # 将样本索引加入其cluster对应的列表中
        clusters[group_id].append(idx)

    # 将每个cluster对应的样本索引列表打乱
    for _, cluster in clusters.items():
        np.random.shuffle(cluster)


    # 记录来自每个cluster的client的样本数量
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64) 

    # 遍历每一个cluster
    for cluster_id in range(n_clusters):
        # 对每个cluster中的每个client赋予一个满足dirichlet分布的权重
        weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
        # np.random.multinomial 表示投掷骰子clusters_sizes[cluster_id]次，落在各client上的权重依次是weights
        # 该函数返回落在各client上各多少次，也就对应着各client应该分得的样本数
        clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    # 对每一个cluster上的每一个client的计数次数进行前缀（累加）求和，
    # 相当于最终返回的是每一个cluster中按照client进行划分的样本分界点下标
    clients_counts = np.cumsum(clients_counts, axis=1)


    def split_list_by_idcs(l, idcs):
        """
        将列表`l` 划分为长度为 `len(idcs)` 的子列表
        第`i`个子列表从下标 `idcs[i]` 到下标`idcs[i+1]`
        （从下标0到下标`idcs[0]`的子列表另算）
        返回一个由多个子列表组成的列表
        """
        res = []
        current_index = 0
        for index in idcs: 
            res.append(l[current_index: index])
            current_index = index

        return res
    
    clients_idcs = [[] for _ in range(n_clients)]
    for cluster_id in range(n_clusters):
        # cluster_split为一个cluster中按照client划分好的样本
        cluster_split = split_list_by_idcs(clusters[cluster_id], clients_counts[cluster_id])

        # 将每一个client的样本累加上去
        for client_id, idcs in enumerate(cluster_split):
            clients_idcs[client_id] += idcs

    return clients_idcs

def pathological_non_iid_split(dataset, n_classes, n_clients, n_classes_per_client):
    # get subset
    data_idcs = list(range(len(dataset)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        _, label = dataset[idx]
        label2index[label].append(idx)

    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]

    def iid_divide(l, g):
            """
            将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
            每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
            返回由不同的groups组成的列表
            """
            num_elems = len(l)
            group_size = int(len(l) / g)
            num_big_groups = num_elems - g * group_size
            num_small_groups = g - num_big_groups
            glist = []
            for i in range(num_small_groups):
                glist.append(l[group_size * i: group_size * (i + 1)])
            bi = group_size * num_small_groups
            group_size += 1
            for i in range(num_big_groups):
                glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
            return glist


    n_shards = n_clients * n_classes_per_client
        # 一共分成n_shards个独立同分布的shards
    shards = iid_divide(sorted_idcs, n_shards)
    np.random.shuffle(shards)
    # 然后再将n_shards拆分为n_client份
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # 这里shard是一个shard的数据索引(一个列表)
            # += shard 实质上是在列表里并入列表
            clients_idcs[client_id] += shard 

    return clients_idcs
    
def rotate_data(client_data, test_data, val_data, n_clients, n_clusters):
    # Next, we simulate a clustering structure in the client population, 
    # by rotating the data of some cluster by a certain degree. 
    # We display 10 data samples from the 1st and the 6th client for illustration.
    if n_clusters > n_clients:
        raise IOError("n_cluster cant be larger than n_clients!")
    
    n_client_per_cluster = n_clients//n_clusters
    n_degree_per_cluster = 360//n_clusters
    n_degree_last = n_degree_per_cluster * (n_clusters - 1)
    # print("n_degree_last: %d" % n_degree_last)
    cnt_cluster = 0
    for i, client_datum in enumerate(client_data):
        if cnt_cluster <= n_clusters - 2:
            degree = n_degree_per_cluster * cnt_cluster 
        else:
            degree = n_degree_last
        # print(degree)
        client_datum.subset_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation((degree, degree)),
                transforms.ToTensor()]
        )
        if (i + 1) % n_client_per_cluster == 0:
            cnt_cluster += 1


    cnt_cluster = 0
    for i, client_datum in enumerate(test_data):
        if cnt_cluster <= n_clusters - 2:
            degree = n_degree_per_cluster * cnt_cluster 
        else:
            degree = n_degree_last
        client_datum.subset_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation((degree,degree)),
                transforms.ToTensor()]
        )
        if (i + 1)% n_client_per_cluster == 0:
            cnt_cluster += 1        
                   

    cnt_cluster = 0
    for i, client_datum in enumerate(val_data):
        if cnt_cluster <= n_clusters - 2:
            degree = n_degree_per_cluster * cnt_cluster 
        else:
            degree = n_degree_last
        client_datum.subset_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation((degree, degree)),
                transforms.ToTensor()]
        )
        if (i + 1) % n_client_per_cluster == 0:
            cnt_cluster += 1  





