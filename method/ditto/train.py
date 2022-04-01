'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-26 20:18:23
LastEditors: ZhangHongYu
LastEditTime: 2022-03-26 20:19:40
'''
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_op(per_model, V, loader, W, epochs=1, lam=0.1):

    lr = 0.1
    def zero_grad(V):
        for v in V.values():
            if v.grad is not None:
                v.grad.data.zero_()

    def ditto_optimize(V, W):
        with torch.no_grad():
            for v, w in zip(V.values(), W.values()):  # 需要更新的模型参数
                v.data -= lr * (v.grad + lam*(v.data - w.data))

    per_model.train()  

    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader: 
            x, y = x.to(device), y.to(device)
            zero_grad(V)

            loss = torch.nn.CrossEntropyLoss()(per_model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]
  
            loss.backward()
            
            ditto_optimize(V, W) 

    return running_loss / samples