# 기본
import os

# 이미지 처리
import numpy as np

# torch Data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# model 
import torch

transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Resize((224,224),antialias=True),
    ])

class SignalDataSet(Dataset):
    def __init__(self, data_dict, class_list, transforms = transform, use_mem = True):
        self.transform = transforms
        self.use_mem = use_mem
        self.data_dict = data_dict
        self.class_list = class_list
        self.data = []
        self.label = []

        for i,cls in enumerate(self.class_list) :
            for path in self.data_dict[cls]:
                if self.use_mem:
                    x = np.load(path)
                    x = x.reshape(-1,3072,1)
                    x = self.transform(x)
                    x = x.view(1,224,-1)
                    self.data.append(x)

                else:
                    self.data.append(path)

                y = torch.zeros(len(self.class_list))
                y[i] = 1

                self.label.append(y)
                    
    def __len__(self):
        result = 0
        for cls in self.class_list:
            result += len(self.data_dict[cls])
        return result

    def __getitem__(self, idx):
        if self.use_mem:
            x = self.data[idx]

        else:
            x = self.data[idx]
            x = np.load(x)
            x = x.reshape(-1,3072,1)
            x = self.transform(x)
            x = x.view(1,224,-1)
            
        y = self.label[idx]

        return x, y

if __name__ == "__main__":
    import json

    with open('config/custom.json', 'r') as file:
        data = json.load(file)

    data_set = SignalDataSet(data['train'],data['class'])
    print(data_set[1][0].shape)