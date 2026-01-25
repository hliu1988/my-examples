import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16*30*30, 10)  # 假设输入是3x32x32

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
