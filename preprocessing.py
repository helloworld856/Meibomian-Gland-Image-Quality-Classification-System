import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from configuration import config


class LightweightEnhancement:
    """
    轻量级增强版本（速度更快，适合训练时实时使用）
    只使用CLAHE增强，训练速度更快
    """
    def __init__(self, clahe_clip_limit=2.0, clahe_tile_size=(8, 8)):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
    
    def __call__(self, img):
        # PIL Image转换为numpy数组
        img_np = np.array(img)
        
        # 转换为BGR
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img_np
        
        # CLAHE增强
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 转换回RGB
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img_rgb)


# 定义预处理（使用CLAHE轻量级增强 + 极强数据增强防止过拟合）
train_transforms = transforms.Compose([
    LightweightEnhancement(clahe_clip_limit=3.0),
    transforms.Resize(config.input_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),  # 进一步增加旋转
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # 更强仿射变换
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.08),  # 更强颜色变化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2))  # 增加擦除概率和范围
])

# 验证集使用相同的增强
val_transforms = transforms.Compose([
    LightweightEnhancement(clahe_clip_limit=3.0),  # 与训练集一致
    transforms.Resize(config.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])