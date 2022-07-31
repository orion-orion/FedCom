'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-26 18:50:20
LastEditors: ZhangHongYu
LastEditTime: 2022-03-26 19:52:39
'''
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in loader: 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            #自带softmax和默认mean
            # 对于lstm模型
            # model(x):(128, 100, 80)
            # y: (128, 80)
            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  

    return running_loss / samples
      
      
def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)
            # predicted: (128, 80)
            if len(y.shape) == 2:
                # next ch pred处理任务
                samples += y.shape[0] * y.shape[1]
            else:
                samples += y.shape[0]
                
            correct += (predicted == y).sum().item()

    # 可能部分client生成的样本数为0
    if samples == 0:
        return -1
    else:        
        return correct/samples