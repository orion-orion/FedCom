'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-12-23 18:53:12
LastEditors: ZhangHongYu
LastEditTime: 2022-04-01 20:22:41
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
from method.my.graph_cluster.community_louvain import best_partition, generate_dendrogram, partition_at_level
from mpl_toolkits.mplot3d import Axes3D
import math

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]

def display_data_distribution(client_idcs, train_labels, num_cls, n_clients, classes):
    # 展示不同client的不同label的数据分布，注意列表命名为clients
    plt.figure(figsize=(20,3))
    plt.hist([train_labels[idc]for idc in client_idcs], stacked=True, 
             bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(n_clients)], rwidth=0.5)
    plt.xticks(np.arange(num_cls), classes)
    plt.legend()
    plt.show()


def display_train_stats(cfl_stats, eps_1, eps_2, communication_rounds):
    # clear_output(wait=True)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    
    # 剔除部分测试样本为0的client
    filtered_acc_clients = [ acc for acc in cfl_stats.acc_clients if acc !=-1]

    acc_mean = np.mean(filtered_acc_clients, axis=1)
    acc_std = np.std(filtered_acc_clients, axis=1)

    plt.fill_between(cfl_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(cfl_stats.rounds, acc_mean, color="C0")
    print("  第 %d 轮的client平均精度为：%f" % (cfl_stats.rounds[-1], acc_mean[-1]))

    if "split" in cfl_stats.__dict__ : 
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")


    if "partition" in cfl_stats.__dict__: 
        for p in cfl_stats.partition:
            plt.axvline(x=p, linestyle="-", color="k", label=r"Partition")


    plt.text(x=communication_rounds, y=1, ha="right", va="top", 
             s="Clusters: {}".format([list(x) for x in cfl_stats.clusters[-1]]))
    
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)
    plt.grid()
    plt.subplot(1,2,2)
    
    plt.plot(cfl_stats.rounds, cfl_stats.mean_norm, color="C1", label=r"$\|\sum_i\Delta W_i \|$")
    plt.plot(cfl_stats.rounds, cfl_stats.max_norm, color="C2", label=r"$\max_i\|\Delta W_i \|$")
    
    plt.axhline(y=eps_1, linestyle="--", color="k", label=r"$\varepsilon_1$")
    plt.axhline(y=eps_2, linestyle=":", color="k", label=r"$\varepsilon_2$")

    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")

    if "partition" in cfl_stats.__dict__:
        for p in cfl_stats.partition:
            plt.axvline(x=p, linestyle="-", color="k", label=r"Partition")


    plt.xlabel("Communication Rounds")
    plt.legend()

    plt.xlim(0, communication_rounds)
    #plt.ylim(0, 2)
    
    plt.grid()
    plt.savefig(fname=os.path.join(os.curdir, 'result_pic', 'result.png'))
    #plt.show()
    

def display_sample(clients, classes): 
    # 对数据集进行可视化
    for client in [clients[0], clients[9]]:
        x, y = iter(client.train_loader).next()

        print("Client {}:".format(client.id))
        plt.figure(figsize=(15,1))
        for i in range(10):
            plt.subplot(1,10,i+1)
            plt.imshow(x[i].numpy().T)
            plt.title("Label: {}".format(classes[y[i].item()]))
        plt.show()

def clear_graph_pic_dir():
    path = "result_pic/graph"
    if not os.path.exists(path):
        os.makedirs(path)
    file_list = os.listdir(path)
    for f in file_list:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

def get_weighted_edges(similarities, idc):
    weighted_edges = []
    for i, node_id1 in zip(range(similarities.shape[0]), idc):
        for j, node_id2 in zip(range(similarities.shape[1]), idc):
            # 不包括自环和负权
            if i != j:
                weighted_edges.append((node_id1, node_id2, max(1e-12, similarities[i][j])))
    return weighted_edges

def graph_split_visualization(G, par_idcs, similarities, c_round, c_id):
    fig = plt.figure(figsize=(12,4))
    col = 3
    row = math.ceil(( 1 + len(par_idcs))/col)


    # 绘制划分前的网络结构
    ax = fig.add_subplot(row, col, 1)
    pos=nx.spring_layout(G,iterations=20)
    #以下语句绘制以带宽为线的宽度的图
    nx.draw_networkx_edges(G,pos,width=[float(d['weight']*10) for (u,v,d) in G.edges(data=True)], ax=ax)
    nx.draw_networkx_nodes(G,pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)     

    for c_id, idc in enumerate(par_idcs):
        sub_graph = nx.Graph()
        # graph.add_nodes_from(idc)
        sub_graph.add_weighted_edges_from(get_weighted_edges(similarities[idc][:, idc], idc))
        # 绘制划分后的网络结构
        ax = fig.add_subplot(row, col, c_id + 2)
        pos=nx.spring_layout(sub_graph,iterations=20)
        #以下语句绘制以带宽为线的宽度的图
        nx.draw_networkx_edges(sub_graph,pos,width=[float(d['weight']*10) for (u,v,d) in sub_graph.edges(data=True)], ax=ax)
        nx.draw_networkx_nodes(sub_graph,pos, ax=ax)
        nx.draw_networkx_labels(sub_graph, pos, ax=ax)
        

    path = "result_pic/graph"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join("result_pic/graph", "%d round %dth cluster split.png" % (c_round, c_id)))