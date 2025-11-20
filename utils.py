"""
工具函数模块
提供日志、文件操作等通用功能
"""
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import json


def setup_logger(
    name: str = "image_quality_classifier",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
    
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def load_json(file_path: str, default: Optional[Dict] = None) -> Dict:
    """
    安全加载JSON文件
    
    Args:
        file_path: JSON文件路径
        default: 默认值（如果文件不存在）
    
    Returns:
        加载的字典数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        if default is not None:
            return default
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析错误: {file_path} - {str(e)}")


def save_json(data: Dict, file_path: str) -> None:
    """
    安全保存JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dir(dir_path: str) -> None:
    """
    确保目录存在
    
    Args:
        dir_path: 目录路径
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_best_model_name(model_info_path: str = './model_save/model_info.txt') -> str:
    """
    从模型信息文件中获取最佳模型名称
    
    Args:
        model_info_path: 模型信息文件路径
    
    Returns:
        最佳模型名称
    """
    import re
    try:
        with open(model_info_path, 'r', encoding='utf-8') as f:
            text = f.readlines()
        
        result = []
        for line in text:
            matches = re.findall(r"(\w+): accuracy (.[.\w]+).+f1 score ([.\w]+)", line)
            if matches:
                result.append(matches[0])
        
        if not result:
            raise ValueError("未找到有效的模型信息")
        
        result.sort(key=lambda x: (float(x[1]) + float(x[2])) / 2, reverse=True)
        return result[0][0]
    except FileNotFoundError:
        raise FileNotFoundError(f"模型信息文件不存在: {model_info_path}")
    except Exception as e:
        raise RuntimeError(f"读取模型信息失败: {str(e)}")

