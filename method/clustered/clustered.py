'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-02-17 11:20:36
LastEditors: ZhangHongYu
LastEditTime: 2022-04-09 18:36:29
'''
import numpy as np
from tqdm import tqdm
from utils.plots import display_train_stats, clear_graph_pic_dir

def clustered_fl(args, clients, server, cfl_stats):

    EPS_1 = 0.4
    EPS_2 = 1.6   

    cluster_indices = [[client_id for client_id in range(len(clients))]]
    client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

    similarities = server.compute_pairwise_similarities(clients)
    server.cache_cluster(cluster_indices, 0)
    #print(similarities)
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
        cluster_indices_new = []

        for idc in cluster_indices:

            # 这里idc是某个簇内client的id集合
            # 计算某个簇内的client的最大参数变化率delta W的范数
            
            max_norm = server.compute_max_update_norm([clients[i] for i in idc])
            # 计算某个簇内的client的平均参数范数
            mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
            if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2 and c_round>20:     
                server.cache_model(idc, clients[idc[0]].W, acc_clients)
                # 最多只能划分为两个簇啊-.-
                # 这里用idc将对应簇的client筛选出来
                # similarities[idc][:,idc]为
                # idc对应的client和除了idc之外的client之间的权重组成的切片矩阵
                # 相当于原本属于idc对应簇的client分裂成了c1，c2两个簇
                # 只是个把戏，其实等价于similarities[idc][idc]
                c1, c2 = server.cluster_clients(similarities[idc][:,idc]) 
                # print("已进行新的簇分裂!!!")
                cluster_indices_new += [list(np.array(idc)[c1]), list(np.array(idc)[c2])]
                cfl_stats.log({"split" : c_round})
                # similarities = np.where(similarities<0, 0, 1)
                # np.fill_diagonal(similarities, 0)
                # draw_communities(similarities, cluster_indices_new, c_round)
            else:  
                # 这里是将当前簇中client的id嵌套在一 d个列表里再合并进去
                # [[1, 2]] + [[5, 6]] = [[1, 2], [5, 6]]
                # 相当于不再分裂，idc簇仍然是idc簇
                cluster_indices_new += [idc]
                
        if len(cluster_indices_new) > len(cluster_indices):
            server.cache_cluster(cluster_indices_new, c_round)
            
        cluster_indices = cluster_indices_new

        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(client_clusters)

        acc_clients = [client.evaluate() for client in clients]
        
        cfl_stats.log({"acc_clients" : acc_clients, "mean_norm" : mean_norm, "max_norm" : max_norm,
                    "rounds" : c_round, "clusters" : cluster_indices})
        
        display_train_stats(cfl_stats, EPS_1, EPS_2, args.n_rounds)
        pbar.update(1)
    for idc in cluster_indices:    
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

