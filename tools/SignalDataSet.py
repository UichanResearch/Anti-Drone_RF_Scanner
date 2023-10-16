# 기본
import os

# 이미지 처리
import numpy as np
from PIL import Image
import random
random.seed(42)

# torch Data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# model 
import torch

transform = transforms.Compose([  
    transforms.ToTensor(),  
    transforms.Resize((224,224)),
    ])

class SignalDataSet(Dataset):
    
    def __init__(self, path, transforms = transform):
        self.path = path
        self.transform = transforms
        self.label = os.listdir(path)
        self.index = np.arange(0,len(self.label))
        self.data = []

        for i in range(len(self.label)):
            cls_name = self.label[i]
            cls_num = i
            target_cls_path = os.path.join(path, cls_name)
            for imgs in os.listdir(target_cls_path):
                inst = {}
                inst["cls_name"] = cls_name
                inst["cls_num"] = cls_num
                inst["data_path"] = os.path.join(target_cls_path,imgs)
                self.data.append(inst)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # data
        x = self.data[idx]["data_path"]
        x = np.load(x)
        x = x.reshape(50,3072,1)
        x = self.transform(x)
        
        # label
        y = torch.zeros(len(self.label))
        y[self.data[idx]["cls_num"]] = 1
        
        return x, y

if __name__ == "__main__":
    a = SignalDataSet("Data")
