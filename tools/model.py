import torch
import torch.nn as nn

class CNN_custom(nn.Module):
    def __init__(self,class_num):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1,16,5), # 244 -> 240
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 240 -> 120
            nn.Conv2d(16,32,5), # 120 -> 116
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 116 -> 58
            nn.Conv2d(32,64,5), # 58 -> 54
            nn.ReLU(),
            nn.MaxPool2d(2,2), # 54 -> 27
        )

        self.fc = nn.Sequential(
            nn.Linear(36864, 27*27),
            nn.ReLU(),
            nn.Linear(27*27, 100),
            nn.ReLU(),
            nn.Linear(100, class_num),
        )

    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(out.size(0), -1) # 참고 view는 reshape과 역할 비슷
        out = self.fc(out)
        return out