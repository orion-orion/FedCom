'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-02-17 11:20:36
LastEditors: ZhangHongYu
LastEditTime: 2022-03-26 20:55:10
'''
from utils.plots import display_train_stats
from tqdm import tqdm

def fed_avg_fl(args, clients, server, cfl_stats):

    EPS_1 = 0.4
    EPS_2 = 1.6   

    cluster_indices = [[client_id for client_id in range(len(clients))]]
        
    # similarities = server.compute_pairwise_similarities(clients)
    #print(similarities)
    # 进行n_rounds轮迭代
    acc_clients = []
    pbar = tqdm(total=args.n_rounds)
    
    
    for c_round in range(1, args.n_rounds+1):
        # print(cluster_indices)
    # c_round为当前的通信轮数
        # 初始时需要保证客户端和服务器模型同步
        if c_round == 1:
            for client in clients:
                client.synchronize_with_server(server)
                
        participating_clients = server.select_clients(clients, frac=1.0)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=args.local_epochs)
            client.reset()
        
        
        max_norm = server.compute_max_update_norm(clients)
        # 计算某个簇内的client的平均参数范数
        mean_norm = server.compute_mean_update_norm(clients)
        
        
        server.aggregate_weight_updates(clients)
    
        for client in clients:
            client.synchronize_with_server(server)    
        

        acc_clients = [client.evaluate() for client in clients]
        
        cfl_stats.log({"acc_clients" : acc_clients, "mean_norm" : mean_norm, "max_norm" : max_norm,
                    "rounds" : c_round, "clusters" : cluster_indices})
        
        display_train_stats(cfl_stats, EPS_1, EPS_2, args.n_rounds)
        pbar.update(1)
        
    for idc in cluster_indices:    
        server.cache_model(idc, clients[idc[0]].W, acc_clients)

