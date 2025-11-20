from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from configuration import config
from torchvision.datasets import ImageFolder
from torch import nn, optim
import torch
import numpy as np
from collections import Counter
from models import EfficientNetCBAM
from preprocessing import train_transforms, val_transforms
from train_validation import train_val_model
from evaluate import evaluate_model
from utils import setup_logger, save_json

logger = setup_logger("train")

def get_labels_from_image_folder(dataset):
    """从数据集中提取所有标签"""
    return [label for _, label in dataset]

def get_simple_optimizer(model, learning_rate=config.learning_rate):
    """优化器设置，添加L2正则化防止过拟合"""
    # 只区分骨干网络和分类器
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:  # 只考虑可训练参数
            if 'features' in name or 'backbone' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

    # 创建优化器，增加weight_decay（L2正则化）防止过拟合
    # 降低学习率 + 更强L2正则化
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.08, 'weight_decay': 2e-4},  # 降低学习率，增强正则化
        {'params': classifier_params, 'lr': learning_rate * 0.8, 'weight_decay': 2e-3}  # 分类器也降低学习率
    ])

    return optimizer


def main():
    # 加载数据集
    train_dataset = ImageFolder(root=config.data_path, transform=train_transforms)
    val_dataset = ImageFolder(root=config.data_path, transform=val_transforms)
    
    # 获取类别映射（供预测时使用）
    res = train_dataset.class_to_idx
    logger.info(f"类别映射: {res}")
    
    # 获取所有样本标签和索引
    all_labels = get_labels_from_image_folder(train_dataset)
    all_indices = list(range(len(train_dataset)))

    logger.info(f"数据集总样本数: {len(train_dataset)}")
    logger.info(f"类别分布: {dict(Counter(all_labels))}")

    # 使用5折交叉验证
    n_split = 5
    # 分层k折交叉，将数据集分成k折，且每一折里的各类别比例与原数据集一致
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

    # 存放每一折训练好的模型名及其f1指标和准确率，并写入文件
    tips = []

    # 存放每一折的结果
    fold_results = []

    # 进行验证
    for fold, (train_indices, val_indices) in enumerate(skf.split(all_indices, all_labels)):
        logger.info(f"\n{'='*50}")
        logger.info(f'训练第{fold+1}折')

        # 创建训练集和验证集
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        # 检查每一折的分布
        train_labels = [all_labels[i] for i in train_indices]
        val_labels = [all_labels[i] for i in val_indices]

        logger.info(f"训练集大小: {len(train_subset)}, 类别分布: {dict(Counter(train_labels))}")
        logger.info(f"验证集大小: {len(val_subset)}, 类别分布: {dict(Counter(val_labels))}")

        # 创建数据加载器
        train_dataloader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, 
                                       num_workers=config.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_subset, batch_size=config.batch_size, 
                                     num_workers=config.num_workers, pin_memory=True)

        # 初始化模型，每一折都重新初始化保证从0开始训练
        model = EfficientNetCBAM(len(config.classNames), True)


        # 定义损失函数和优化器（使用类别权重处理不平衡）
        # 计算类别权重：样本少的类别权重大
        class_counts = [train_labels.count(i) for i in range(len(config.classNames))]
        class_weights = torch.tensor([1.0 / count for count in class_counts], dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(config.classNames)  # 归一化
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))
        
        optimizer = get_simple_optimizer(model)

        # 学习率调度（使用CosineAnnealing，更平滑）
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

        # 训练模型,返回的是最高准确率和F1指数
        accuracy, f1 = train_val_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_fold=fold+1
        )

        logger.info(f'最高准确率: {accuracy:.4f}, 最高F1分数: {f1:.4f}')
        tips.append(f'best_model_{fold+1}: accuracy {accuracy:.4f},  f1 score {f1:.4f}')
        fold_results.append(accuracy)

    #  计算交叉验证的总体结果
    logger.info(f"\n{'='*60}")
    logger.info(f"{n_split}折交叉验证最终结果:")
    logger.info(f"各折准确率: {[f'{acc:.4f}' for acc in fold_results]}")
    logger.info(f"平均准确率: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    logger.info(f"最佳准确率: {np.max(fold_results):.4f} (第 {np.argmax(fold_results)+1} 折)")
    logger.info(f"最差准确率: {np.min(fold_results):.4f} (第 {np.argmin(fold_results)+1} 折)")

    # 保存模型信息和类别映射
    with open(config.model_info_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tips))
    logger.info(f"模型信息已保存: {config.model_info_path}")
    
    # 保存类别映射供预测使用
    save_json(res, str(config.class_mapping_path))
    logger.info(f"类别映射已保存: {config.class_mapping_path}")

    # 测试模型
    # 加载测试数据集
    test_dataset = ImageFolder(config.test_data_path, transform=val_transforms)
    test_data_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    evaluate_model(test_data_loader)

if __name__ == '__main__':
    main()


