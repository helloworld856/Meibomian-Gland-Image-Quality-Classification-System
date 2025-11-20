import torch
import platform
import os
from pathlib import Path

class Config:
    """项目配置类"""
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 项目根目录
    root_dir = Path(__file__).parent
    
    # 模型加载路径
    load_model_path = './model_save/'
    model_save_dir = root_dir / 'model_save'
    
    # 类别名称
    classNames = ['normal', 'reflection', 'blur', 'incomplete exposure']
    num_classes = len(classNames)
    
    # 早停配置
    early_stopping_patience = 10  # 若连续10轮验证集的F1指标和准确率没有提升，则停止
    
    # 输入图像尺寸
    input_size = (384, 800)  # (height, width)
    
    # 训练配置
    num_epochs = 40
    batch_size = 16
    data_path = './data/train_val'    # 数据集路径
    test_data_path = './data/test'    # 测试集路径
    learning_rate = 1e-3
    
    # DataLoader配置（Windows上num_workers>0可能有问题）
    num_workers = 0 if platform.system() == 'Windows' else 2
    pin_memory = True if torch.cuda.is_available() else False
    
    # 混合精度训练
    use_amp = True  # 是否使用自动混合精度训练
    
    # 类别映射文件路径
    class_mapping_path = model_save_dir / 'class_mapping.json'
    model_info_path = model_save_dir / 'model_info.txt'
    
    def __init__(self):
        """初始化配置并创建必要的目录"""
        # 初始化后创建必要的目录
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        (self.model_save_dir / 'best_model_info').mkdir(parents=True, exist_ok=True)

# 创建配置实例
config = Config()
