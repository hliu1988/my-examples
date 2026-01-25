import torch
import torch.nn as nn

# 定义一个简单的模型


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型和输入
model = SimpleNet()
x = torch.randn(32, 10)  # batch_size=32, input_dim=10

# 使用 torch.compile 编译模型（默认使用 inductor 后端）
compiled_model = torch.compile(model)

# 首次运行会触发编译（可能稍慢），后续运行会更快
output = compiled_model(x)
print(output.shape)  # 应该输出 torch.Size([32, 1])
