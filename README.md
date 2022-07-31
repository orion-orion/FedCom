<!--
 * @Descripttion: 
 * @Version: 1.0
 * @Author: ZhangHongYu
 * @Date: 2021-12-23 18:53:12
 * @LastEditors: ZhangHongYu
 * @LastEditTime: 2022-04-01 20:27:20
-->
# 基于社区检测的多任务聚类联邦学习
[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/TipDMCup20)                                                                        [![](https://img.shields.io/github/license/orion-orion/TipDMCup21)](https://github.com/orion-orion/TipDMCup21/LICENSE)                                                                           [![](https://img.shields.io/github/stars/orion-orion/TipDMcup21?style=social)](https://github.com/orion-orion/TipDMCup21)                                                                        [![](https://img.shields.io/github/issues/orion-orion/TipDMCup21)](https://github.com/orion-orion/TipDMCup21/issues)

## 1 简介
本项目为SWPU本科毕业设计《基于社区检测的多任务聚类联邦学习》。本研究提出了一种多任务聚类联邦学习的新方法，该方法的特点是基于社区检测来进行聚类簇的动态划分。本研究的方法规避了任务簇由人工指定的弊端，从而在算法后期对前期错误划分进行动态修正，缓解了因前期划分失误带来的负面影响。在三个不同数据集上进行的实验表明，相比本文对比的其它几个联邦学习方法，本文提出的基于动态划分的簇状联邦多任务学习方法在大多数情况下表现更好。在大多数情况下，本文所提出的算法划分更加准确，达到的准确率更高，收敛速度更快。其中于本文的算法具有动态调整聚簇的特性，所以即使在最初的几轮迭代中划分错误，也能在后面的迭代中对错误的聚簇划分进行调整，从而取得更佳的精度表现。

## 2 算法流程描述

### 2.1 算法整体流程
整体算法流程分为构建带权无向图、任务簇动态划分、任务簇参数更新三个部分，如下图所示：

<img src="paper_pic/算法流程图.png" width="800" height="350">

### 2.2 算法分步描述

#### 2.2.1 构建带权无向图
本研究将$n$个客户端节点视为带权无图$\mathcal{G}=(\mathcal{V},\mathcal{E})$, 这里$\mathcal{V}$代表各客户端节点，$\mathcal{E}$代表各客户端节点之间的交互。节点$k$的邻居为$\mathcal{N}_{k}=\{l \in \mathcal{V} \mid(l, k) \in \mathcal{E}\}$我们设$a_{kl}$为节点$l$和节点$k$之间的边权。
<img src="paper_pic/6节点的所有邻居.png" width="250" height="200">

在实际运行环境下，预先并不知道任务之间的关系，便无法预先为边赋权。本研究认为节点数据分布相似性（边权）可以由模型学习到。可以证明分布相似性和参数间的余弦距离相关<sup>[1]</sup>：
$$
a_{k l}=\frac{\left\langle\Delta \theta_{k}, \Delta \theta_{l}\right\rangle}{\left\|\Delta \theta_{k}\right\|\left\|\Delta \theta_{l}\right\|}
$$

#### 2.2.2 聚类簇动态划分算法
首先，将图的一个社团划分和一个聚类簇相对应。然后，采用依据边的权重划分社团的社团划分算法<sup>[2]</sup>。该算法在每一轮迭代更新参数后先进行试探性的社团检测。此外，本研究采用模块度$M$对划分质量进行评估，如果划分后的模块度$M_t$相对于上一轮的模块性$M_{t-1}$取得了$\varepsilon$的增幅，即：$M_{t}-M_{t-1}>\varepsilon$，则采用当前划分。
<img src="paper_pic/聚类簇动态划分示意图.png" width="600" height="190">
本研究采用的Louvain社区检测算法<sup>[2]</sup>是一个采用贪心策略的多次迭代算法，每一次迭代都由划分和重构两部分组成。其中划分步骤只使用局部调整优化模块性，做为初步的簇划分；重构步骤则会聚合当前划分出的聚簇，形成新图用于下一次迭代。

<img src="paper_pic/louvain算法示意图.png" width="600" height="430">

#### 2.2.3 聚类簇参数更新

以任务簇为单位进行参数更新的共享。也即相同任务簇内的节点对参数更新求加权平均，其权重根据各任务节点的样本数量来分配，如下式所示：
$$
\Delta \theta_{k}=\sum_{t \in C_{k}} \frac{n_{t}}{n_{C_{k}}} \Delta \theta_{t}
$$
如下图所示，簇$C_1$中根据节点$1、2、3$的参数更新进行加权平均，簇$C_2$中根据节点$4、5$的参数更新进行加权平均。

<img src="paper_pic/聚类簇参数平均示意图.png" width="400" height="300">

## 3 实验设计

### 3.1 数据集Non-IID划分
采用联邦学习的benckmark数据集：EMNIST、CIFAR10、FEMNIST等数据集<sup>[3]</sup>对模型进行评估，采用Dirichlet分布对数据集进行Non-IID划分。下图是对EMNIST数据集采用Dirichlet分布示意图：

<img src="paper_pic/数据集non-iid划分示意图.png" width="800" height="200">

### 3.2 采用图像翻转法初始化簇结构

假设有10个client，K个初始化簇结构：

- K=2，则client1-5不变，client6-10的数据进行旋转180°。

- K=3，则client1-3不变，client4-6中图片旋转120°，client7-10图片旋转240°。
- 以此类推。

下面是对CIFAR10数据集进行翻转示意图：

<img src="paper_pic/对cifar10数据集进行翻转示意图.png" width="800" height="200">

### 3.3 实验结果
#### 3.3.1 EMNIST数据集
在EMNIST数据集上，10个client，300轮迭代条件下的实验结果示意图：
|  表头   | 精度（k=2）  | 精度（k=3）  | 精度（k=4）  |
|  ----  | ----  |----  |----  |
| Local  | 0.759 |0.759 |0.760 |
| FedAvg  | 0.773 |0.795 |0.776 |
| Clustered | 0.803 |0.787 |0.783 |
| Ditto  | 0.790 |0.795 |0.776|
| Ours  | 0.806 |0.801 |0.787 |

可见模型评估精度在各个$k$值均领先于对比论文TNNLS20的Clustered算法<sup>[4]</sup>和ICML21的Ditto算法<sup>[5]</sup>。
在$k=2$的时候，本研究的算法和TNNLS20的算法都进行了正确的划分，本研究的算法首次进行划分更早，划分结果示意图如下：

<img src="paper_pic/emnist-k=2时和对比方法的比较.png" width="800" height="300">

在$k=3$的时候，本研究的算法一次性就进行了正确的划分，TNNLS20的算法经过了多次划分而且划分错误，划分结果示意图如下：
<img src="paper_pic/emnist-k=3时和对比方法的比较.png" width="800" height="300">

#### 3.3.2 CIFAR10数据集
在CIFAR10数据集上，10个client，300轮迭代条件下的实验结果示意图：
|  表头   | 精度（k=2）  | 精度（k=3）  | 精度（k=4）  |
|  ----  | ----  |----  |----  |
| Local  | 0.721 |0.787 |0.771 |
| FedAvg  | 0.788 |0.796 |0.792 |
| Clustered | 0.800 |0.800 |0.786 |
| Ditto  | 0.796 |0.798 |0.793|
| Ours  | 0.805 |0.799 |0.796 |
可见模型评估精度在多数k值情况均领先于对比论文TNNLS20的Clustered算法和ICML21的Ditto算法。

在$k=2$的时候，本研究的算法那最开始也进行了错误的划分，但后期及时进行了动态的调整，而TNNLS20的算法不能进行动态调，划分结果示意图如下：
<img src="paper_pic/cifar10-k=2时和对比方法的比较.png" width="800" height="270">

#### 3.3.3 FEMNIST数据集

在FEMNIST数据集上，80个client，300轮迭代条件下的实验结果示意图：
|  表头   | 精度（k=2）  | 精度（k=3）  | 精度（k=4）  |
|  ----  | ----  |----  |----  |
| Local  | 0.548 |0.523 |0.553 |
| FedAvg  | 0.750 |0.742 |0.707 |
| Clustered | 0.739 |0.685 |0.665 |
| Ditto  | 0.760 |0.685 |0.709|
| Ours  | 0.764 |0.743 |0.712 |

可见模型评估精度在各个$k$值均领先于对比论文TNNLS20的Clustered算法和ICML21的Ditto算法。

### 环境依赖
运行以下命令安装环境依赖：
```
pip install -r requirements.txt
```

### 使用方法
运行:

```
python main.py CIFAR10 My
```

第一个参数用于选择数据集，选项有：`EMNIST`, `FashionMNIST`, `CIFAR10`，`CIFAR100`, `Shakespeare`；第二个参数用于选择需要验证的方法，选项有：`My`,`Clustered`, `FedAvg`, `Ditto`, `Local`。




## 参考
[1] Sattler F, Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints[J], TNNLS 2020
[2]  Blondel V D etc. Fast unfolding of communities in large networks[J]. Journal of statistical mechanics: theory and exp, 2008
[3] https://github.com/TalwalkarLab/leaf
[4] Sattler F, Müller K R, Samek W. Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints[J]. IEEE transactions on neural networks and learning systems, 2020, 32(8): 3710-3722.
[5] Li T, Hu S, Beirami A, et al. Ditto: Fair and robust federated learning through personalization[C]//International Conference on Machine Learning. PMLR, 2021: 6357-6368.


