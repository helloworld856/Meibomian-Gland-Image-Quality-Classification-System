import os
import json
from pathlib import Path
from models import EfficientNetCBAM
from configuration import config
import torch
from PIL import Image
from preprocessing import val_transforms
from utils import setup_logger, load_json, get_best_model_name
import argparse

logger = setup_logger("predict")

# 加载类别映射
try:
    res = load_json(str(config.class_mapping_path))
    logger.info(f"成功加载类别映射: {res}")
except Exception as e:
    logger.warning(f"加载类别映射失败，使用默认映射: {e}")
    res = {'blur': 0, 'incomplete exposure': 1, 'normal': 2, 'reflection': 3}

# 创建反向映射（从索引到类别名）
idx_to_class = {v: k for k, v in res.items()}

# 加载模型
def load_model(model_path=None):
    """加载模型"""
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
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise

# 默认加载最佳模型
try:
    model = load_model()
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    model = None

predict_transforms = val_transforms


def predict_single_image(image_path: str, model, show_top_k: int = 1):
    """
    预测单张图片
    
    Args:
        image_path: 图片路径
        model: 模型实例
        show_top_k: 显示前k个预测结果
    
    Returns:
        dict: 预测结果
    """
    if model is None:
        raise RuntimeError("模型未加载")
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = predict_transforms(img).unsqueeze(0).to(config.device)
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 获取top-k结果
            top_probs, top_indices = torch.topk(probabilities, k=min(show_top_k, len(res)))
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                class_name = idx_to_class[idx.item()]
                results.append({
                    'class': class_name,
                    'confidence': prob.item(),
                    'index': idx.item()
                })
            
            return results
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到图片文件: {image_path}")
    except Exception as e:
        raise RuntimeError(f"预测失败: {str(e)}")


def predict_batch(folder_path: str, model, show_top_k: int = 1):
    """
    批量预测文件夹中的图片
    
    Args:
        folder_path: 文件夹路径
        model: 模型实例
        show_top_k: 显示前k个预测结果
    
    Returns:
        list: 预测结果列表
    """
    if model is None:
        raise RuntimeError("模型未加载")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"不是有效的文件夹路径: {folder_path}")
    
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    img_list = [
        f for f in os.listdir(folder_path) 
        if Path(f).suffix.lower() in img_extensions
    ]
    
    if not img_list:
        raise ValueError("文件夹中没有找到图片文件")
    
    logger.info(f'找到 {len(img_list)} 张图片，开始预测...')
    results = []
    
    for name in img_list:
        try:
            img_path = os.path.join(folder_path, name)
            pred_results = predict_single_image(img_path, model, show_top_k)
            results.append({
                'filename': name,
                'predictions': pred_results
            })
        except Exception as e:
            logger.error(f'{name}: 预测失败 - {str(e)}')
            results.append({
                'filename': name,
                'error': str(e)
            })
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像质量分类预测')
    parser.add_argument('--model', type=str, default=None, help='模型路径（默认使用最佳模型）')
    parser.add_argument('--image', type=str, default=None, help='单张图片路径')
    parser.add_argument('--folder', type=str, default=None, help='图片文件夹路径')
    parser.add_argument('--top-k', type=int, default=1, help='显示前k个预测结果')
    parser.add_argument('--output', type=str, default=None, help='保存结果到JSON文件')
    
    args = parser.parse_args()
    
    # 如果指定了模型路径，重新加载模型
    if args.model:
        global model
        model = load_model(args.model)
    
    if model is None:
        logger.error("模型未加载，无法进行预测")
        return
    
    results = None
    
    # 命令行模式
    if args.image:
        try:
            pred_results = predict_single_image(args.image, model, args.top_k)
            print(f"\n图片: {args.image}")
            for i, result in enumerate(pred_results, 1):
                print(f"  {i}. {result['class']}: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
            results = {'image': args.image, 'predictions': pred_results}
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return
    
    elif args.folder:
        try:
            results = predict_batch(args.folder, model, args.top_k)
            print(f"\n批量预测结果 ({len(results)} 张图片):")
            for item in results:
                if 'error' in item:
                    print(f"  {item['filename']}: 错误 - {item['error']}")
                else:
                    print(f"  {item['filename']}:")
                    for i, pred in enumerate(item['predictions'], 1):
                        print(f"    {i}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            return
    
    else:
        # 交互模式
        print('\n=== 图像质量分类预测系统 ===')
        while True:
            print('\n请选择模式：')
            print('  1. 单张图片预测')
            print('  2. 批量图片预测（文件夹）')
            print('  3. 退出')
            
            try:
                target = int(input('请输入选项 (1-3): ').strip())
            except ValueError:
                print("输入无效，请输入数字1、2或3")
                continue
            
            if target == 1:
                path = input('请输入图片路径: ').strip().strip('"')
                try:
                    pred_results = predict_single_image(path, model, args.top_k)
                    print(f'\n预测结果:')
                    for i, result in enumerate(pred_results, 1):
                        print(f"  {i}. {result['class']}: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
                except Exception as e:
                    print(f"错误: {e}")
            
            elif target == 2:
                folder_path = input('请输入文件夹路径: ').strip().strip('"')
                try:
                    results = predict_batch(folder_path, model, args.top_k)
                    print(f'\n批量预测结果 ({len(results)} 张图片):')
                    for item in results:
                        if 'error' in item:
                            print(f"  {item['filename']}: 错误 - {item['error']}")
                        else:
                            print(f"  {item['filename']}:")
                            for i, pred in enumerate(item['predictions'], 1):
                                print(f"    {i}. {pred['class']}: {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
                except Exception as e:
                    print(f"错误: {e}")
            
            elif target == 3:
                print('退出程序...')
                break
            else:
                print("无效选项，请重新输入")
    
    # 保存结果
    if args.output and results:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {args.output}")


if __name__ == '__main__':
    main()





