import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from train_eval import train_op, eval_op

device = "cuda" if torch.cuda.is_available() else "cpu"



def copy(target, source):
    for name in target:
        target[name].data = source[name].data.clone()
    
def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()
    
def weighted_reduce_add_average(targets, sources):
    for target in targets:
        for name in target:
            tmp = torch.sum(torch.stack([source[name].data * weight for (source, weight) in sources]), dim=0).clone()
            target[name].data += tmp
        
def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])


  
class Client(object):
    def __init__(self, model_fn, optimizer_fn, train_data, test_data, val_data, idnum, batch_size=128):
        self.model = model_fn().to(device)

        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.n_train_samples = len(train_data)
        
        self.optimizer = optimizer_fn(self.model.parameters())
            
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)

        self.eval_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
        
        self.id = idnum
        
        # 此处为传引用，model参数变化self.W自然变化
        self.W = {key : value for key, value in self.model.named_parameters()}
        
        self.dW = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key : torch.zeros_like(value) for key, value in self.model.named_parameters()}
        
        # weight for aggregate
        self.weight = torch.tensor(0)
        
    def synchronize_with_server(self, server):
        #print("copy前 %d 的权重"  % self.id, "\n", list(self.model.named_parameters()))
        # for param in self.model.parameters():   
        #     param += server.W
        copy(target=self.W, source=server.W)
        #print("copy后 %d 的权重"  % self.id, "\n", list(self.model.named_parameters()))
        # print("copy后 %d " % self.id, self.W)
            
    def compute_weight_update(self, epochs=1, loader=None):
        copy(target=self.W_old, source=self.W)
        self.optimizer.param_groups[0]["lr"]*=0.99
        train_stats = train_op(self.model, self.train_loader if not loader else loader, self.optimizer, epochs)
        subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)
        return train_stats  


    def reset(self): 
        copy(target=self.W, source=self.W_old)
    
    def evaluate(self, loader=None):
        return eval_op(self.model, self.eval_loader if not loader else loader)

    
class Server(object):
    def __init__(self, model_fn):
        self.model = model_fn().to(device)
        self.W = {key : value for key, value in self.model.named_parameters()}
        self.model_cache = []

    def select_clients(self, clients, frac=1.0):
        return random.sample(clients, int(len(clients)*frac)) 
    
    def aggregate_weight_updates(self, clients):
        #print("reduce 前 server 的权重", "\n", list(self.model.named_parameters()))
        weighted_reduce_add_average(targets=[self.W], sources=[(client.dW, client.weight) for client in clients])
        #print("reduce 后 server 的权重", "\n", list(self.model.named_parameters()))
            
            
    def compute_max_update_norm(self, cluster):
        # for client in cluster:
        #     print(client.dW)
        return np.max([torch.norm(flatten(client.dW)).item() for client in cluster])

    
    def compute_mean_update_norm(self, cluster):
        return torch.norm(torch.mean(torch.stack([flatten(client.dW) for client in cluster]), 
                                     dim=0)).item()

    def cache_model(self, idcs, params, accuracies):
        self.model_cache += [(idcs, 
                            {name : params[name].data.clone() for name in params}, 
                            [accuracies[i] for i in idcs])]



