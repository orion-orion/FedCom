'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-02-17 11:20:22
LastEditors: ZhangHongYu
LastEditTime: 2022-04-09 20:23:36
'''

import networkx as nx
from .graph_cluster.community_louvain import best_partition, modularity
from utils.plots import display_train_stats
from utils.plots import display_train_stats, clear_graph_pic_dir, draw_communities 
from tqdm import tqdm
import numpy as np

# import copy
# __MIN = 0.0000001
# 初始化图必须要[(0, 1, 0.3), (1, 2,0.5)]这种元组列表结构
def get_weighted_edges(similarities, idc):
    weighted_edges = []
    for i, node_id1 in zip(range(similarities.shape[0]), idc):
        for j, node_id2 in zip(range(similarities.shape[1]), idc):
            # 不包括自环和负权
            if i != j:
                weighted_edges.append((node_id1, node_id2, max(1e-12, similarities[i][j])))
    return weighted_edges

def my_fl(args, clients, server, cfl_stats):

    EPS_1 = 0.4
    EPS_2 = 1.6   
    #MOD_MIN = 0.45  # 一般在0.3-0.7为较好的社区
    pre_mod = 0.2
    min_val = 0
    cluster_indices = [[client_id for client_id in range(len(clients))]]
        
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    # 初始化图
    # 直接按照初始的client参数计算初始相似度，初始时client参数都为0，图的权重都为0
    similarities = server.compute_pairwise_similarities(clients)
    server.cache_cluster(cluster_indices, 0)

    graph = nx.Graph()

    # 初始化图必须要[(0, 1, 0.3), (1, 2,0.5)]这种元组列表结构
        
    graph.add_weighted_edges_from(get_weighted_edges(similarities, range(len(clients))))
    # print(graph) Graph with 10 nodes and 55 edges，因为是无向图

    # 进行n_rounds轮迭代
    acc_clients = []
    
    clear_graph_pic_dir()
    pbar = tqdm(total=args.n_rounds)
    for c_round in range(1, args.n_rounds+1):
        # print(cluster_indices)

    # c_round为当前的通信轮数
        # 设置客户端初始化模型
        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)
                
        participating_clients = server.select_clients(clients, frac=1.0)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.local_epochs)
            client.reset()

        # 现将表示所有clients之间参数相似度的邻接矩阵算好放在这里
        similarities = server.compute_pairwise_similarities(clients)

        #print(similarities)
        # 更新图的相似度矩阵
        graph.add_weighted_edges_from(get_weighted_edges(similarities, range(len(clients))))

        cluster_indices_new = []

        for idc in cluster_indices:

            # 这里idc是某个簇内client的id集合
            # 计算某个簇内的client的最大参数变化率delta W的范数
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            # 计算某个簇内的client的平均参数范数
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])

        # if c_round > 20:    #20
        par_idcs, par_dict = community_detection_method(graph) 
        cur_mod = modularity(par_dict, graph)
        print(cur_mod)
        if cur_mod - pre_mod > min_val:
            server.cache_model(idc, clients[idc[0]].W, acc_clients)
            cfl_stats.log({"partition" : c_round})

            if args.n_clients >= 20 or args.n_clusters == 2:
                similarities = np.where(similarities<0, 0, 1)
                np.fill_diagonal(similarities, 0)
                draw_communities(similarities, par_idcs, c_round)

            cluster_indices_new += par_idcs
            pre_mod = cur_mod
            server.cache_cluster(cluster_indices_new, c_round)
        else:
            cluster_indices_new  = cluster_indices
        # else:
        #     cluster_indices_new = cluster_indices


        cluster_indices = cluster_indices_new

        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate() for client in clients]
        
        cfl_stats.log({"acc_clients" : acc_clients, "mean_norm" : mean_norm, "max_norm" : max_norm,
                    "rounds" : c_round, "clusters" : cluster_indices})
        pbar.update(1)
        display_train_stats(cfl_stats, EPS_1, EPS_2, args.n_rounds)
    for idc in cluster_indices:    
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

def community_detection_method(graph:nx.Graph):

    par_dict = best_partition(graph, resolution= graph.number_of_nodes())
 
    # part为字典，如{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    # 我们需要将其k，v反一下，我们要的是一个cluster有哪些client
    cluster_to_clients = {}
    for client_id, cluster_id in par_dict.items():
        if cluster_id not in cluster_to_clients.keys():
            cluster_to_clients[cluster_id] = [client_id]
        else:
            cluster_to_clients[cluster_id].append(client_id)
    
    cluster_indices_new = [] # 这里面重新定义了就把参数屏蔽了，必须要return，不能再修改参数了
    for c_clients in cluster_to_clients.values():
        cluster_indices_new.append(c_clients)

    return cluster_indices_new, par_dict
