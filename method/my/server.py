'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-26 20:06:12
LastEditors: ZhangHongYu
LastEditTime: 2022-04-09 17:20:03
'''
from fl_devices import Server, flatten, weighted_reduce_add_average
import torch

class CommunityServer(Server):
    def __init__(self, model_fn):
        super().__init__(model_fn)
        self.cluster_cache = []
        self.r_cache = []

    def compute_pairwise_similarities(self, clients):
        return pairwise_angles([client.dW for client in clients])
        
    def aggregate_clusterwise(self, client_clusters):
        weights_sum  = [0 for i in range(len(client_clusters))] 
        for cluster_id, cluster in enumerate(client_clusters):
            cluster_weight_sum = sum([client.weight for client in cluster])
            weights_sum[cluster_id] = cluster_weight_sum

            
        for cluster_id, cluster in enumerate(client_clusters):
            weighted_reduce_add_average(targets=[client.W for client in cluster], 
                               sources=[(client.dW, client.weight/weights_sum[cluster_id]) for client in cluster])


    def cache_cluster(self, cluster_idcs, c_round):
        self.cluster_cache.append(cluster_idcs)
        self.r_cache.append(c_round)
    


# 逐(i,j)对计算参数相似度
# 这里的sources为所有client的参数的变化量delta_W组成的列表
def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten(source1) # 多维参数要用flatten压成一维的
            s2 = flatten(source2)
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)
    return angles.numpy()