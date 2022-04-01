'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-12-23 18:53:12
LastEditors: ZhangHongYu
LastEditTime: 2022-02-24 10:22:49
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import string

class ConvNet(torch.nn.Module):
    def __init__(self, input_size, channels, num_classes):
        super(ConvNet, self).__init__()
        # 但
        self.conv1 = torch.nn.Conv2d(channels, 32, 5) #输入通道，输出通道，卷积核大小
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        
        if input_size == 28:
            self.fc1 = torch.nn.Linear(1024, 2048) 
            self.input_size = 28
            self.output = torch.nn.Linear(2048, num_classes) #62

        elif input_size == 32:
            self.fc1 = torch.nn.Linear(64 * 5 * 5, 2048) # 10
            self.input_size = 32
            self.output = torch.nn.Linear(2048, num_classes) #10


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.input_size == 28:
            x = x.view(-1, 1024)
        elif self.input_size == 32:
            x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class MobileNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features,num_classes)

    def forward(self, x):
        return self.model(x)



def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


class NextCharacterLSTM(nn.Module):
    def __init__(
        self, 
        input_size=len(string.printable), 
        embed_size=8, 
        hidden_size=256, 
        output_size=len(string.printable), 
        n_layers=2):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        self.rnn.flatten_parameters()
        encoded = self.encoder(input_) # (128, 80, 8)
        output, _ = self.rnn(encoded) # (128, 80, 256)
        output = self.decoder(output) # (128, 80, 100)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        # (128, 100, 80)
        return output

