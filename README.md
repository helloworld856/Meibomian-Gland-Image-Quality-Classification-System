### 注意：当前项目未提供具体数据集，但提供在数据集上的训练和评估结果及相应的模型，可以运行后续的predict.py和gui.py。

# 睑板腺图像质量分类项目

基于深度学习的图像质量分类系统，用于识别睑板腺图像质量问题。

## 功能特点

- 🎯 **四分类任务**：识别正常、反射、模糊、曝光不足四种睑板腺图像质量问题
- 🧠 **先进模型**：EfficientNet-B0 + CBAM注意力机制
- 📊 **5折交叉验证**：确保模型泛化能力
- 🚀 **训练优化**：混合精度训练、早停机制、学习率调度
- 📈 **完整评估**：混淆矩阵、ROC曲线、详细分类报告

## 项目结构

```
project/
├── configuration.py      # 配置文件
├── models.py             # 模型定义（EfficientNetCBAM + CBAM）
├── preprocessing.py      # 数据预处理和增强
├── train.py             # 主训练脚本（5折交叉验证）
├── train_validation.py   # 训练和验证逻辑
├── evaluate.py          # 模型评估
├── predict.py           # 单张/批量图片预测
├── export_model.py      # 模型导出（ONNX/TorchScript）
├── gui.py              # GUI图形界面
├── utils.py             # 工具函数（日志、文件操作等）
├── requirements.txt     # 依赖包列表
├── README.md            # 项目文档
├── data/                # 数据目录
│    ├── train_val/       # 训练/验证数据
│    │       ├── blur 
│    │       ├── incomplete exposure
│    │       ├── normal
│    │       └── reflection
│    └── test/            # 测试数据
│          ├── blur 
│          ├── incomplete exposure
│          ├── normal
│          └── reflection
└── model_save/          # 保存的模型文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程会进行5折交叉验证，自动保存最佳模型。

### 2. 评估模型

```bash
python evaluate.py
```

评估测试集性能，生成混淆矩阵和ROC曲线。也可以指定特定模型：

```bash
python evaluate.py --model ./model_save/best_model_1.pth
```

### 3. 预测图片

**交互模式：**
```bash
python predict.py
```

**命令行模式：**
```bash
# 单张图片
python predict.py --image path/to/image.jpg

# 批量预测
python predict.py --folder path/to/images/ --top-k 3

# 保存结果到JSON
python predict.py --folder path/to/images/ --output results.json
```

### 4. 导出模型

**导出为ONNX格式：**
```bash
python export_model.py --format onnx
```

**导出为TorchScript格式：**
```bash
python export_model.py --format torchscript
```

**同时导出两种格式：**
```bash
python export_model.py --format both
```

**指定模型和输出路径：**
```bash
python export_model.py --model ./model_save/best_model_1.pth --output model.onnx
```

### 5. GUI界面（图形化预测）

启动图形界面进行预测：

```bash
python gui.py
```

**功能特点：**
- 📷 **单张图片预测**：选择单张图片，显示图片预览和Top-3预测结果
- 📁 **批量预测**：选择图片文件夹，批量预测所有图片
- 🎨 **直观界面**：左侧显示图片预览，右侧显示预测结果
- 📊 **详细结果**：显示类别名称、置信度和百分比

**使用说明：**
1. 点击"选择单张图片"按钮，选择要预测的图片
2. 或点击"选择图片文件夹"按钮，选择包含图片的文件夹
3. 预测结果会显示在右侧文本框中

## 配置说明

主要配置在 `configuration.py` 中：

- `input_size`: 输入图像尺寸 (384, 800)
- `batch_size`: 批次大小 16
- `num_epochs`: 训练轮数 40
- `learning_rate`: 学习率 1e-3
- `early_stopping_patience`: 早停耐心值 10

## 模型架构

- **骨干网络**: EfficientNet-B0 (ImageNet预训练)
- **注意力机制**: CBAM (通道注意力 + 空间注意力)
- **分类器**: 自定义全连接层 + 高Dropout防过拟合

## 训练策略

- ✅ 5折分层交叉验证
- ✅ 类别权重处理不平衡数据
- ✅ 数据增强（CLAHE、随机翻转、旋转、颜色抖动等）
- ✅ 梯度裁剪防止梯度爆炸
- ✅ 余弦退火学习率调度
- ✅ 早停机制防止过拟合
- ✅ 混合精度训练加速（AMP）
- ✅ 完整的日志系统

## 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1-score
- AUC-ROC

## 注意事项

- Windows系统上 `num_workers` 设置为0以避免多进程问题
- 确保有足够的GPU内存（建议至少4GB）
- 训练时间取决于数据量和硬件配置

## 许可证

MIT License

