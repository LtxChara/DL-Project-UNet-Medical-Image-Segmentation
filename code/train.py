import os
import glob
import copy
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

from HMT_unet.hmt_unet import HMTUNet
# 从 model.py 中导入所有网络模型
from model import UNet, UNetPlusPlus

# -----------------------------------------------------------------
# 1. 配置中心: 所有的超参数和路径都在这里设置
# -----------------------------------------------------------------
def get_config():
    """
    返回一个包含所有实验配置的字典。
    'experiments_to_run' 是一个列表，其中每个元素都是一个独立的实验设置。
    """
    config = {
        "data_path": "",
        "model_save_path": "./models",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "img_size": 256,
        "k_folds": 5,
        "epochs": 100,
        "batch_size": 8,
        "optimizer": "AdamW",
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "lr_scheduler": "CosineAnnealingLR",
        "early_stopping_patience": 30,
    }

    # --- 定义所有要进行的对比实验 ---
    config["experiments_to_run"] = [
        {
            "experiment_name": "HMTUNet_with_Augmentation",
            "model_name": "HMTUNet",
            "use_augmentation": True,
        },
        {
            "experiment_name": "UNet_without_Augmentation",
            "model_name": "HMTUNet",
            "use_augmentation": False,
        },

    ]
    return config

# -----------------------------------------------------------------
# 2. 数据增强 (Augmentation) 与数据集定义 (Dataset)
# -----------------------------------------------------------------
def get_transforms(img_size, augment=True):
    # 验证集/测试集总是使用不带增强的变换
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    if augment:
        # 训练集使用的数据增强流水线
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.0625, 0.0625), rotate=(-45, 45), p=0.5),
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(p=0.5),
            A.GaussNoise(p=0.2),
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        # 如果不使用增强，训练集的变换与验证集相同
        train_transform = val_transform
    
    return train_transform, val_transform


class ISICDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Albumentations 需要读入 NumPy 数组格式
        # 使用 OpenCV 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 直接以灰度图读取

        if self.transform:
            # 应用增强变换
            # Albumentations 会自动处理图像和掩码的对应关系
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 确保掩码是二值的 (0.0 或 1.0)，并增加一个通道维度
        mask = (mask > 0.5).float().unsqueeze(0)
        
        return image, mask

# -----------------------------------------------------------------
# 3. 损失函数 (Loss) 与评估指标 (Metrics)
# -----------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs是模型的输出logits, 需要先经过sigmoid
        inputs = torch.sigmoid(inputs)       
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def dice_coefficient(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# -----------------------------------------------------------------
# 4. 训练与验证的核心逻辑
# -----------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, epoch_info):
    model.train()
    running_loss = 0.0
    # 使用tqdm包装loader，创建进度条
    pbar = tqdm(loader, desc=f"Training {epoch_info}", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, list): # For UNet++ deep supervision
            loss = 0
            for output in outputs:
                loss += criterion(output, masks)
            loss /= len(outputs)
        else:
            loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        running_loss += batch_loss * images.size(0)
        # 在进度条后实时显示当前批次的损失值
        pbar.set_postfix(loss=f"{batch_loss:.4f}")
        
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device, epoch_info):
    model.eval()
    total_dice = 0.0
    # 为验证过程也添加进度条
    pbar = tqdm(loader, desc=f"Validation {epoch_info}", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            dice = dice_coefficient(outputs, masks)
            total_dice += dice.item()
    return total_dice / len(loader)

# -----------------------------------------------------------------
# 5. 主执行函数
# -----------------------------------------------------------------
def main():
    config = get_config()
    os.makedirs(config["model_save_path"], exist_ok=True)
    
    all_train_images = sorted(glob.glob(os.path.join(config["data_path"], "train", "images", "*.jpg")))
    all_train_masks = sorted(glob.glob(os.path.join(config["data_path"], "train", "masks", "*.png")))
    
    kfold = KFold(n_splits=config["k_folds"], shuffle=True, random_state=42)
    
    available_models = {"UNet": UNet, "UNetPlusPlus": UNetPlusPlus, "HMTUNet": HMTUNet}
        
    # --- 遍历所有要运行的实验 ---
    for experiment in config["experiments_to_run"]:
        exp_name = experiment["experiment_name"]
        model_name = experiment["model_name"]
        use_aug = experiment["use_augmentation"]

        if model_name not in available_models:
            print(f"警告: 实验 '{exp_name}' 指定的模型 '{model_name}' 不可用，将跳过。")
            continue
            
        print(f"\n{'='*25} 开始实验: {exp_name} {'='*25}")
        
        fold_results = []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(all_train_images)):
            print(f"\n----- 第 {fold+1}/{config['k_folds']} 折 -----")
            
            train_img_paths = [all_train_images[i] for i in train_ids]
            train_mask_paths = [all_train_masks[i] for i in train_ids]
            val_img_paths = [all_train_images[i] for i in val_ids]
            val_mask_paths = [all_train_masks[i] for i in val_ids]
            
            # 根据实验配置决定是否使用数据增强
            train_transform, val_transform = get_transforms(config["img_size"], augment=use_aug)
            
            train_dataset = ISICDataset(train_img_paths, train_mask_paths, transform=train_transform)
            val_dataset = ISICDataset(val_img_paths, val_mask_paths, transform=val_transform)
            
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
            
            model = available_models[model_name]().to(config["device"])
            
            criterion = lambda pred, target: 0.5 * nn.BCEWithLogitsLoss()(pred, target) + 0.5 * DiceLoss()(pred, target)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
            
            best_val_dice = 0.0
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
            
            for epoch in range(1, config["epochs"] + 1):
                epoch_info = f"[{exp_name} | Fold {fold+1}/{config['k_folds']} | Epoch {epoch}/{config['epochs']}]"
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, config["device"], epoch_info)
                val_dice = evaluate(model, val_loader, config["device"], epoch_info)
                scheduler.step()
                
                print(f"Epoch {epoch:03d} Summary | Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")
                
                if val_dice > best_val_dice:
                    print(f"  \033[92mValidation Dice Improved ({best_val_dice:.4f} --> {val_dice:.4f}). Saving Model...\033[0m")
                    best_val_dice = val_dice
                    best_model_weights = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= config["early_stopping_patience"]:
                    print(f"\n\033[93mEarly stopping triggered after {config['early_stopping_patience']} epochs with no improvement.\033[0m")
                    break
            
            print(f"Fold {fold+1} finished. Best Validation Dice: {best_val_dice:.4f}")
            fold_results.append(best_val_dice)
            
            # 保存模型时使用实验名称以作区分
            save_path = os.path.join(config["model_save_path"], f"{exp_name}_fold{fold+1}_best.pth")
            torch.save(best_model_weights, save_path)
            print(f"Best model for this fold saved to: {save_path}")
            
        avg_dice = np.mean(fold_results)
        std_dice = np.std(fold_results)
        print(f"\n{'='*25} {exp_name} K-Fold Cross-Validation Summary {'='*25}")
        print(f"{config['k_folds']}-Fold Average Dice Score: {avg_dice:.4f} ± {std_dice:.4f}")

if __name__ == "__main__":
    main()