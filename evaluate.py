from configuration import config
import torch
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.preprocessing import label_binarize
from torchvision.datasets import ImageFolder
from preprocessing import val_transforms
from torch.utils.data import DataLoader
from models import EfficientNetCBAM
from utils import setup_logger, load_json, get_best_model_name
import os

logger = setup_logger("evaluate")


def evaluate_model(dataloader, model_path=None):
    """
    评估模型性能
    
    Args:
        dataloader: 测试数据加载器
        model_path: 模型路径（默认使用最佳模型）
    """
    logger.info('开始评估模型...')
    
    # 加载类别映射
    try:
        res = load_json(str(config.class_mapping_path))
        logger.info(f"成功加载类别映射: {res}")
    except Exception as e:
        logger.warning(f"加载类别映射失败，使用默认映射: {e}")
        res = {'blur': 0, 'incomplete exposure': 1, 'normal': 2, 'reflection': 3}

    # 加载模型
    if model_path is None:
        try:
            best_model_name = get_best_model_name()
            model_path = config.model_save_dir / f'{best_model_name}.pth'
            logger.info(f"自动选择最佳模型: {best_model_name}")
        except Exception as e:
            logger.error(f"无法自动选择模型: {e}")
            raise
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = EfficientNetCBAM()
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=config.device))
        model.to(config.device)
        model.eval()
        logger.info(f"成功加载模型: {model_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

    print('测试集结果：')
    # 保存预测标签、真实标签和预测概率
    y_pred = []
    y_label = []
    y_prob = []  # 用于保存预测概率

    logger.info('开始预测...')
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(config.device)
            labels = labels.to(config.device)

            outputs = model(images)

            # 获取预测概率（使用softmax）
            probabilities = torch.softmax(outputs, dim=1)
            y_prob.extend(probabilities.cpu().numpy())

            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_label.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f'已处理 {batch_idx + 1}/{len(dataloader)} 批次')

    # 转换为numpy数组
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 获取类别名称（按索引排序）
    idx_to_class = {v: k for k, v in res.items()}
    class_names = [idx_to_class[i] for i in range(len(res))]
    n_classes = len(class_names)

    logger.info(f"类别数量: {n_classes}, 类别: {class_names}")
    
    # 1. 输出详细分类报告（包含每类的准确率、F1-score等）
    print("\n" + "="*60)
    print("分类报告:")
    print("="*60)
    print(classification_report(y_label, y_pred, target_names=class_names, digits=4))

    # 2. 绘制混淆矩阵
    cm = confusion_matrix(y_label, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = config.model_save_dir / 'best_model_info' / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    logger.info(f"混淆矩阵已保存: {cm_path}")
    plt.close()

    # 3. 计算并绘制ROC曲线
    # 将标签二值化（用于多类ROC曲线）
    y_label_bin = label_binarize(y_label, classes=range(n_classes))

    # 计算每个类别的ROC曲线和AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown'][:n_classes]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.4f})')

    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)
    roc_path = config.model_save_dir / 'best_model_info' / 'roc_curves_model.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    logger.info(f"ROC曲线已保存: {roc_path}")
    plt.close()

    # 4. 输出每类的详细指标（四分类）
    print("\n" + "="*60)
    print("每类详细指标（四分类）:")
    print("="*60)

    # 从分类报告中提取指标（四分类）
    report_multi = classification_report(
        y_label,
        y_pred,
        target_names=class_names,
        output_dict=True,
        digits=4,
    )

    tips = ''
    for class_name in class_names:
        class_data = report_multi[class_name]

        # 计算该类准确率（该类被正确分类的比例）
        class_mask = y_label == class_names.index(class_name)
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(y_pred[class_mask] == y_label[class_mask])
            acc = f"  准确率 (Accuracy): {class_accuracy:.4f}\n\n"
        else:
            acc = f"  准确率 (Accuracy): 0.0000 (无样本)\n\n"

        tips += (
            f"{class_name}:\n"
            f"  精确率 (Precision): {class_data['precision']:.4f}\n"
            f"  召回率 (Recall): {class_data['recall']:.4f}\n"
            f"  F1-score: {class_data['f1-score']:.4f}\n"
            f"  AUC: {roc_auc[class_names.index(class_name)]:.4f}\n"
            + acc
        )

    # 输出总体准确率（四分类）
    overall_accuracy = report_multi['accuracy']
    print(f"总体准确率（四分类）: {overall_accuracy:.4f}")

    # 输出宏平均和加权平均F1-score（四分类）
    print(f"宏平均F1-score（四分类）: {report_multi['macro avg']['f1-score']:.4f}")
    print(f"加权平均F1-score（四分类）: {report_multi['weighted avg']['f1-score']:.4f}")

    tips += (
        f"\n总体准确率（四分类）: {overall_accuracy:.4f}\n"
        f"宏平均F1-score（四分类）: {report_multi['macro avg']['f1-score']:.4f}\n"
        f"加权平均F1-score（四分类）: {report_multi['weighted avg']['f1-score']:.4f}\n"
    )

    # 5. 基于四分类结果构造二分类：normal vs abnormal（非正常三类合并）
    print("\n" + "="*60)
    print("二分类结果（normal vs abnormal）:")
    print("="*60)

    try:
        normal_idx = class_names.index('normal')
    except ValueError:
        logger.warning("未在类别中找到 'normal'，跳过二分类评估")
    else:
        # 二分类标签：0 = normal, 1 = abnormal
        y_binary = (y_label != normal_idx).astype(int)

        # 二分类概率：使用“异常”概率 = 1 - P(normal)
        prob_normal = y_prob[:, normal_idx]
        prob_abnormal = 1.0 - prob_normal

        # ROC 与 AUC（正类为 abnormal）
        fpr_bin, tpr_bin, _ = roc_curve(y_binary, prob_abnormal)
        auc_bin = auc(fpr_bin, tpr_bin)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_bin, tpr_bin, color='red', lw=2,
                 label=f'ROC curve (AUC = {auc_bin:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Binary ROC Curve (normal vs abnormal)')
        plt.legend(loc="lower right")
        plt.grid(True)
        bin_roc_path = config.model_save_dir / 'best_model_info' / 'roc_curve_binary.png'
        plt.savefig(bin_roc_path, dpi=300, bbox_inches='tight')
        logger.info(f"二分类 ROC 曲线已保存: {bin_roc_path}")
        plt.close()

        # 基于阈值 0.5 的二分类预测
        y_binary_pred = (prob_abnormal >= 0.5).astype(int)

        # 二分类报告
        bin_report = classification_report(
            y_binary,
            y_binary_pred,
            target_names=['normal', 'abnormal'],
            output_dict=True,
            digits=4,
        )

        # 打印二分类报告
        print(classification_report(
            y_binary,
            y_binary_pred,
            target_names=['normal', 'abnormal'],
            digits=4,
        ))

        # 二分类混淆矩阵
        cm_bin = confusion_matrix(y_binary, y_binary_pred)
        print("二分类混淆矩阵 (0=normal, 1=abnormal):")
        print(cm_bin)

        # 绘制并保存二分类混淆矩阵
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm_bin,
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=['normal', 'abnormal'],
            yticklabels=['normal', 'abnormal'],
            cbar_kws={'label': 'Count'},
        )
        plt.title('Binary Confusion Matrix (normal vs abnormal)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_bin_path = config.model_save_dir / 'best_model_info' / 'confusion_matrix_binary.png'
        plt.savefig(cm_bin_path, dpi=300, bbox_inches='tight')
        logger.info(f"二分类混淆矩阵已保存: {cm_bin_path}")
        plt.close()

        # 将二分类结果写入 tips
        tips += "\n二分类结果（normal vs abnormal）:\n"

        for label_name in ['normal', 'abnormal']:
            class_data = bin_report[label_name]
            tips += (
                f"{label_name}:\n"
                f"  精确率 (Precision): {class_data['precision']:.4f}\n"
                f"  召回率 (Recall): {class_data['recall']:.4f}\n"
                f"  F1-score: {class_data['f1-score']:.4f}\n"
            )

        tips += (
            f"  AUC (abnormal 作为正类): {auc_bin:.4f}\n"
            f"  二分类总体准确率: {bin_report['accuracy']:.4f}\n"
            f"  宏平均F1-score（二分类）: {bin_report['macro avg']['f1-score']:.4f}\n"
            f"  加权平均F1-score（二分类）: {bin_report['weighted avg']['f1-score']:.4f}\n"
        )

    info_path = config.model_save_dir / 'best_model_info' / 'best_model_info.txt'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(tips)
    logger.info(f"评估信息已保存: {info_path}")
    
    print("\n" + "="*60)
    print("评估完成！")
    print("="*60)

