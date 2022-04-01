'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2022-03-13 21:27:39
LastEditors: ZhangHongYu
LastEditTime: 2022-03-20 16:01:34
'''
from torch.utils.data import Subset
class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform

    def __getitem__(self, idx):

        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y   

    def __len__(self):
        return len(self.indices)
    