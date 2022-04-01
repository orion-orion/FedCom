'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-26 20:07:41
LastEditors: ZhangHongYu
LastEditTime: 2022-03-26 20:17:40
'''
from fl_devices import Client
from train_eval import eval_op

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from .train import train_op

class DittoClient(Client):
    def __init__(self, model_fn, optimizer_fn, train_data, test_data, val_data, idnum, batch_size=128):
        super().__init__(model_fn, optimizer_fn, train_data, test_data, val_data, idnum, batch_size)
        # this model is used for ditto
        self.per_model = model_fn().to(device)
        # V (personalized weight)is used for the ditto method 
        self.V = {key : value for key, value in self.per_model.named_parameters()}

    #this method is used for the method of dito
    def compute_personalized_weight(self, epochs=1, loader=None, lam=0.1):
        train_stats = train_op(self.per_model, self.V, \
            self.train_loader if not loader else loader, self.W_old, epochs=epochs, lam=lam)
        
    # used in ditto to evaluete the personalized model 
    def evaluate_per(self, loader=None):
        return eval_op(self.per_model, self.eval_loader if not loader else loader)
    


