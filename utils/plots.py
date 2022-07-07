'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-12-23 18:53:12
LastEditors: ZhangHongYu
LastEditTime: 2022-04-25 15:13:26
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np
from method.my.graph_cluster.community_louvain import best_partition, generate_dendrogram, partition_at_level
# Standard Library
import random
from collections import defaultdict
from copy import copy
import os

# Third Party
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import cm
from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
import pandas as pd

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


    # if len(filtered_acc_clients) <=20:  #客户端数目小于20，将直接展示客户端划分情况
    # text = "Clusters: {}".format([list(x) for x in cfl_stats.clusters[-1]])
    # wtxt = WrapText(x=communication_rounds, y=1, text=text, width=420, ha="left", va='top',size = 8,\
    #     horizontalalignment='left',bbox=dict(boxstyle='square,pad=1', fc='black', ec='none')) #500
    
    # WrapText(x=communication_rounds, y=1, ha="right", va="top", 
    #         s=, clip_on=False)
    #ax.add_artist(wtxt)

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




def _inter_community_edges(G, partition):
    edges = defaultdict(list)

    for (i, j) in G.edges():
        c_i = partition[i]
        c_j = partition[j]

        if c_i == c_j:
            continue

        edges[(c_i, c_j)].append((i, j))

    return edges


def _position_communities(G, partition, **kwargs):
    hypergraph = nx.Graph()
    hypergraph.add_nodes_from(set(partition))

    inter_community_edges = _inter_community_edges(G, partition)
    for (c_i, c_j), edges in inter_community_edges.items():
        hypergraph.add_edge(c_i, c_j, weight=len(edges))

    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # Set node positions to positions of its community
    pos = dict()
    for node, community in enumerate(partition):
        pos[node] = pos_communities[community]

    return pos


def _position_nodes(G, partition, **kwargs):
    communities = defaultdict(list)
    for node, community in enumerate(partition):
        communities[community].append(node)

    pos = dict()
    for c_i, nodes in communities.items():
        subgraph = G.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


# Adapted from: https://stackoverflow.com/questions/43541376/how-to-draw-communities-with-networkx
def community_layout(G, partition):
    pos_communities = _position_communities(G, partition, scale=7.0)  # 10.0
    pos_nodes = _position_nodes(G, partition, scale=2.0) # 2.0

    # Combine positions
    pos = dict()
    for node in G.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


#########
# PATCHES
#########


def _node_coordinates(nodes):
    collection = copy(nodes)
    collection.set_offset_position("data")
    return collection.get_offsets()


def _convex_hull_vertices(node_coordinates, community):
    # community: [0, 1, 2] 说明这个社区只有0,1,2好节点
    # node_coordinates: shape(10, 2)，有正有负的坐标
    
    points = np.array(node_coordinates[list(community)])
    # points: shape(3,2) 从 node_coordinates 选出来的3个点

    hull = ConvexHull(points)
    # array([3.10922335, 2.6718853 ])
    # array([6.57332497, 2.54976051])
    # array([4.94703733, 5.61082291])
    
    x, y = points[hull.vertices, 0], points[hull.vertices, 1]
    vertices = np.column_stack((x, y))
    # vertices: shape(3,2)
    # array([3.10922335, 2.6718853 ])
    # array([6.57332497, 2.54976051])
    # array([4.94703733, 5.61082291])


    return vertices


# https://en.wikipedia.org/wiki/Shoelace_formula#Statement
def _convex_hull_area(vertices):
    A = 0.0
    for i in range(-1, vertices.shape[0] - 1):
        A += vertices[i][0] * (vertices[i + 1][1] - vertices[i - 1][1])

    return A / 2


# https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
def _convex_hull_centroid(vertices):
    A = _convex_hull_area(vertices)

    c_x, c_y = 0.0, 0.0
    for i in range(vertices.shape[0]):
        x_i, y_i = vertices[i]
        if i == vertices.shape[0] - 1:
            x_i1, y_i1 = vertices[0]
        else:
            x_i1, y_i1 = vertices[i + 1]

        cross = ((x_i * y_i1) - (x_i1 * y_i))

        c_x += (x_i + x_i1) * cross
        c_y += (y_i + y_i1) * cross

    return c_x / (6 * A), c_y / (6 * A)


def _scale_convex_hull(vertices, offset):
    c_x, c_y = _convex_hull_centroid(vertices)
    for i, vertex in enumerate(vertices):
        v_x, v_y = vertex

        if v_x > c_x:
            vertices[i][0] += offset
        else:
            vertices[i][0] -= offset
        if v_y > c_y:
            vertices[i][1] += offset
        else:
            vertices[i][1] -= offset

    return vertices


def _community_patch(vertices):
    vertices = _scale_convex_hull(vertices, 1) # TODO: Make offset dynamic
    # vertices: shape(3, 2)
    # vertices.T 两个维度是一样的
    
    # if vertices.shape[0] == 3:
    vertices = np.concatenate([vertices, vertices[-1].reshape(1, -1)], axis=0)
    
    tck, u = splprep(vertices.T, u=None, s=0.0, per=1, k=3)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    path = Path(np.column_stack((x_new, y_new)))
    patch = PathPatch(path, alpha=0.50, linewidth=0.0)
    return patch


def draw_community_patches(nodes, communities, axes):
    node_coordinates = _node_coordinates(nodes)
    vertex_sets = []
    for c_i, community in enumerate(communities):
        vertices = _convex_hull_vertices(node_coordinates, community)
        patch = _community_patch(vertices)
        patch.set_facecolor(nodes.to_rgba(c_i))

        axes.add_patch(patch)
        vertex_sets.append(patch.get_path().vertices)

    _vertices = np.concatenate(vertex_sets)
    xlim = [_vertices[:, 0].min(), _vertices[:, 0].max()]
    ylim = [_vertices[:, 1].min(), _vertices[: ,1].max()]

    axes.set_xlim(xlim)
    axes.set_ylim(ylim)


##################
# DRAW COMMUNITIES
##################


def draw_communities(adj_matrix, communities, c_round, dark=False, filename=None, dpi=None, seed=1):

    np.random.seed(seed)
    random.seed(seed)

    G = nx.from_numpy_matrix(adj_matrix)
    partition = [0 for _ in range(G.number_of_nodes())]
    for c_i, nodes in enumerate(communities):
        for i in nodes:
            partition[i] = c_i

    plt.rcParams["figure.facecolor"] = "black" if dark else "white"
    plt.rcParams["axes.facecolor"] = "black" if dark else "white"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    node_size = 10200 / G.number_of_nodes()
    linewidths = 34 / G.number_of_nodes() #34

    pos = community_layout(G, partition)
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_color=partition,
        linewidths=linewidths,
        cmap=cm.jet,
        ax=ax,
        node_size=node_size
    )
    nodes.set_edgecolor("w")
    edges = nx.draw_networkx_edges(
        G,
        pos=pos,
        edge_color=(1.0, 1.0, 1.0, 0.75) if dark else (0.6, 0.6, 0.6, 1.0),
        width=linewidths,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_color="white", font_size=node_size*0.02)
    draw_community_patches(nodes, communities, ax)


    path = "result_pic/graph"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join("result_pic/graph", "%d round cluster partition.png" % (c_round)),\
        bbox_inches = 'tight')
    
    
def draw_result_table(args, clients, server):
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
    
    
    if args.method == "Clustered" or args.method == "My":
        frame = pd.DataFrame({"Cluster":server.cluster_cache}, index=\
            ["Round {}".format(r) for r in server.r_cache])
        frame = frame.style.set_properties(**{'background-color': 'white', "align":"center"})
        frame.to_html(os.path.join(path, "cluster_log.html"))