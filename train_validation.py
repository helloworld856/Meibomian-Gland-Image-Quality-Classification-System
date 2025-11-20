import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from configuration import config
from torch import nn
from torch.amp import autocast, GradScaler
from utils import setup_logger
from pathlib import Path

logger = setup_logger("train_validation")


# 训练函数，返回平均f1指标和准确率
def train_val_model(
    model, 
    train_dataloader, 
    val_dataloader, 
    optimizer, 
    criterion, 
    scheduler, 
    num_fold, 
    epoches=config.num_epochs
):
    """
    训练和验证模型
    
    Args:
        model: 模型实例
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        scheduler: 学习率调度器
        num_fold: 当前折数
        epoches: 训练轮数
    
    Returns:
        tuple: (最高准确率, 最高F1分数)
    """
    model.to(config.device)
    
    # 混合精度训练
    scaler = GradScaler(device='cuda') if config.use_amp and config.device.type == 'cuda' else None
    
    # 早停
    patience = config.early_stopping_patience
    highest_accuracy = 0
    highest_f1_score = 0


    for epoch in range(epoches):
        # 训练阶段
        print(f'\rEpoch {epoch + 1}/{config.num_epochs}:')

        model.train()
        # 记录一轮训练的总损失、准确样本数和总样本数
        train_loss = 0
        correct = 0
        total = 0

        # 包裹迭代对象，显示进度条
        progress_bar = tqdm.tqdm(
            train_dataloader,
            desc='train_phase',
            leave=True,
            ncols=100
        )

        for i, j in progress_bar:
            # 移动到相应设备
            images, labels = i.to(config.device), j.to(config.device)

            # 清零梯度
            optimizer.zero_grad()

            # 混合精度训练
            if scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 前向传播
                outputs = model(images)
                # 计算每个批次所有样本的平均损失
                loss = criterion(outputs, labels)
                # 反向传播和更新参数
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 累加每个批次的平均损失
            train_loss += loss.item()

            # 累加正确样本数和总样本数
            _, predicted = torch.max(outputs.data, 1)  # 找到每个样本的最大值的索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })

        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_dataloader)  # 所有样本的平均损失
        avg_train_accuracy = correct / total  # 准确率

        logger.info(f'Epoch {epoch + 1}/{epoches} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}')

        # 验证阶段
        model.eval()

        # 收集所有预测和真实标签，用于计算每轮的精确率、召回率和F1指标
        all_preds = []
        all_labels = []

        # 记录一轮训练的总损失、准确样本数和总样本数
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_progress = tqdm.tqdm(
                val_dataloader,
                f'val_phase',
                leave=True,
            )

            for i, j in val_progress:
                val_images, val_labels = i.to(config.device), j.to(config.device)

                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_labels)
                val_loss += loss.item()

                _, val_predicted = torch.max(val_outputs, 1)
                all_preds.extend(val_predicted.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())
                val_total += val_labels.size(0)
                val_correct += (val_predicted==val_labels).sum().item()

            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_accuracy = val_correct / val_total

            # 计算精确率、召回率和f1指标
            val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_labels, all_preds, average='weighted')
            
            logger.info(
                f'Epoch {epoch + 1}/{epoches} - Val Loss: {avg_val_loss:.4f}, '
                f'Val Acc: {avg_val_accuracy:.4f}, Precision: {val_precision:.4f}, '
                f'Recall: {val_recall:.4f}, F1: {val_f1:.4f}'
            )

            # 判断是否早停
            if avg_val_accuracy > highest_accuracy or val_f1 > highest_f1_score:
                # 只要有一个指标提升就保存
                if avg_val_accuracy > highest_accuracy:
                    highest_accuracy = avg_val_accuracy
                if val_f1 > highest_f1_score:
                    highest_f1_score = val_f1
                patience = config.early_stopping_patience
                # 保存模型
                model_save_path = config.model_save_dir / f'best_model_{num_fold}.pth'
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'模型已保存: {model_save_path}')
            else:
                patience -= 1

            if patience == 0:
                logger.info('触发早停机制！')
                break

        # 更新学习率
        scheduler.step()

    logger.info(f"训练完成！最高准确率: {highest_accuracy:.4f}, 最高F1分数: {highest_f1_score:.4f}")
    return highest_accuracy, highest_f1_score

