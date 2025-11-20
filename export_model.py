"""
模型导出工具
将训练好的PyTorch模型导出为ONNX格式，便于部署
"""
import torch
import argparse
from pathlib import Path
from models import EfficientNetCBAM
from configuration import config
from utils import setup_logger, get_best_model_name
import os

logger = setup_logger("export_model")


def export_to_onnx(
    model_path: str = None,
    output_path: str = None,
    input_size: tuple = None,
    opset_version: int = 13
):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model_path: 模型文件路径（默认使用最佳模型）
        output_path: 输出ONNX文件路径
        input_size: 输入图像尺寸 (height, width)
        opset_version: ONNX opset版本
    """
    if input_size is None:
        input_size = config.input_size
    
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
    
    if output_path is None:
        output_path = config.model_save_dir / 'model.onnx'
    else:
        output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = EfficientNetCBAM(num_classes=config.num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    logger.info(f"开始导出ONNX模型...")
    logger.info(f"输入尺寸: {input_size}")
    logger.info(f"输出路径: {output_path}")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"✓ ONNX模型导出成功: {output_path}")
        
        # 验证导出的模型
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("✓ ONNX模型验证通过")
        except ImportError:
            logger.warning("未安装onnx库，跳过模型验证")
        except Exception as e:
            logger.warning(f"ONNX模型验证失败: {e}")
        
        return str(output_path)
    except Exception as e:
        logger.error(f"导出ONNX模型失败: {e}")
        raise


def export_to_torchscript(model_path: str = None, output_path: str = None, input_size: tuple = None):
    """
    将PyTorch模型导出为TorchScript格式
    
    Args:
        model_path: 模型文件路径
        output_path: 输出文件路径
        input_size: 输入图像尺寸
    """
    if input_size is None:
        input_size = config.input_size
    
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
    
    if output_path is None:
        output_path = config.model_save_dir / 'model.pt'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = EfficientNetCBAM(num_classes=config.num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    logger.info(f"开始导出TorchScript模型...")
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(output_path))
        logger.info(f"✓ TorchScript模型导出成功: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"导出TorchScript模型失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='导出模型为ONNX或TorchScript格式')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径（默认使用最佳模型）')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'], 
                       default='onnx', help='导出格式')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset版本')
    parser.add_argument('--input-size', type=int, nargs=2, default=None, 
                       metavar=('HEIGHT', 'WIDTH'), help='输入图像尺寸')
    
    args = parser.parse_args()
    
    input_size = tuple(args.input_size) if args.input_size else config.input_size
    
    try:
        if args.format in ['onnx', 'both']:
            export_to_onnx(
                model_path=args.model,
                output_path=args.output if args.format == 'onnx' else None,
                input_size=input_size,
                opset_version=args.opset
            )
        
        if args.format in ['torchscript', 'both']:
            torchscript_output = args.output if args.format == 'torchscript' else None
            if torchscript_output and args.format == 'both':
                # 如果同时导出两种格式，需要为TorchScript指定不同的输出路径
                torchscript_output = Path(args.output).with_suffix('.pt')
            export_to_torchscript(
                model_path=args.model,
                output_path=torchscript_output,
                input_size=input_size
            )
        
        logger.info("模型导出完成！")
    except Exception as e:
        logger.error(f"导出失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

