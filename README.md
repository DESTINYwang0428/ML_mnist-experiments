# ML_mnist-experiments

基于 PyTorch 的 MNIST 手写数字识别实验，对比不同模型结构、激活函数的效果。

---

## 📊 实验结果

| 版本 | 模型结构 | 激活函数 | 参数量 | 训练准确率 | 测试准确率 | 备注 |
|------|----------|---------|--------|-----------|-----------|------|
| v1.0 | 1层Conv + 3层FC | ReLU | ~40万 | 99.77% | 98.72% | 基线 |
| v1.1 | 3层Conv + 2层FC | ReLU | ~8万 | 99.26% | 98.80% | 加深卷积，泛化更好 |
| v1.2 | 1层Conv(128核,5×5) + 4层FC | ReLU | ~2600万 | 99.73% | 98.79% | 大模型，轻微过拟合 |
| **v1.3** | **3层Conv + 2层FC** | **tanh** | **~8万** | **99.42%** | **99.11%** | **🏆 最佳测试准确率** |

---

## 🔍 主要结论

1. **激活函数**：在 MNIST 上，`tanh` 的泛化能力优于 `ReLU`（测试集 99.11% vs 98.80%）。  
2. **模型容量**：MNIST 不需要大模型，~8万参数足够，堆参数只会增加过拟合风险。  
3. **最佳模型**：`v1.3`（3层Conv + tanh + 2层FC）参数量最小，测试准确率最高。

---

## 📁 项目结构

```
├── 机器学习/
│   ├── 代码栈/
│   │   ├── DL_only1conv.py          # v1.0 基线
│   │   ├── DL_ChangeFunction.py     # v1.3 tanh 版本
│   │   └── DL_differentLR.py        # v1.2 大模型
│   ├── 损失函数.md
│   ├── 激活函数.md
│   └── 优化器.md
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/DESTINYwang0428/ML_mnist-experiments.git
   cd ML_mnist-experiments
   ```

2. **安装依赖**（Python 3.8+）
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

3. **运行实验**
   ```bash
   python "机器学习/代码栈/DL_ChangeFunction.py"
   ```

---

## 📦 依赖环境

```txt
Python 3.8+
torch >= 2.0.0
torchvision >= 0.15.0
matplotlib >= 3.7.0
numpy >= 1.24.0
```

---

## 📌 License

MIT License（详见 [LICENSE](LICENSE) 文件）

---

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系。
