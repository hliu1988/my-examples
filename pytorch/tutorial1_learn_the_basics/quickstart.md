```bash
$ ls -lrt data/FashionMNIST/raw
total 83840
-rw-rw-r-- 1 hliu hliu 26421880  1月 24 20:59 train-images-idx3-ubyte.gz
-rw-rw-r-- 1 hliu hliu 47040016  1月 24 20:59 train-images-idx3-ubyte
-rw-rw-r-- 1 hliu hliu    29515  1月 24 20:59 train-labels-idx1-ubyte.gz
-rw-rw-r-- 1 hliu hliu    60008  1月 24 20:59 train-labels-idx1-ubyte
-rw-rw-r-- 1 hliu hliu  4422102  1月 24 21:00 t10k-images-idx3-ubyte.gz
-rw-rw-r-- 1 hliu hliu  7840016  1月 24 21:00 t10k-images-idx3-ubyte
-rw-rw-r-- 1 hliu hliu     5148  1月 24 21:00 t10k-labels-idx1-ubyte.gz
-rw-rw-r-- 1 hliu hliu    10008  1月 24 21:00 t10k-labels-idx1-ubyte
```

在这个 FashionMNIST 数据集里，`training data` 和 `test data` 是深度学习任务中标准的数据集划分，它们的作用和对应文件如下：

### 1. **training data（训练数据）**
- **对应文件**：以 `train-` 开头的文件（`train-images-idx3-ubyte`、`train-labels-idx1-ubyte` 及它们的压缩包）
- **内容**：包含 60,000 张 28×28 的灰度服装图像，以及对应的 60,000 个标签（每个标签是 0-9 的整数，对应 T 恤、裤子、鞋子等服装类别）。
- **作用**：用于训练模型，让模型学习图像特征与标签之间的映射关系，是模型“学习知识”的基础。

### 2. **test data（测试数据）**
- **对应文件**：以 `t10k-` 开头的文件（`t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte` 及它们的压缩包）
- **内容**：包含 10,000 张 28×28 的灰度服装图像，以及对应的 10,000 个标签。
- **作用**：用于评估训练好的模型性能，这些数据在训练过程中从未被模型见过，用来验证模型对新数据的泛化能力（比如准确率、召回率等指标），避免模型“过拟合”。

### 关键对应关系
- `t10k-` 是 `test 10000` 的缩写，直接对应测试集的 10,000 个样本。
- 训练集（60,000 样本）和测试集（10,000 样本）的划分是 FashionMNIST 数据集的标准设定，和 MNIST 手写数字数据集的划分完全一致。
