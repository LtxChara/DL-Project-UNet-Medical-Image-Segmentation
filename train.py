import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unet_base import UNet, center_crop
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
torch.backends.cudnn.benchmark = True
# ----------------------------
# 1. 数据集定义
# ----------------------------
class ISICDataset(Dataset):
    """
    加载 ISIC 2017 预处理后的图像 (.jpg) 与掩码 (.png)，
    并返回 tensor 格式：(img, mask)
    """
    def __init__(self, img_dir, mask_dir, img_transform=None, mask_transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir,  "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        assert len(self.img_paths) == len(self.mask_paths), \
            "图像与掩码数量不匹配"
        self.img_transform  = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 读取
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # 2. 变换（resize, ToTensor, normalize）
        if self.img_transform:
            img = self.img_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # 3. 二值化掩码 (0 or 1)
        mask = (mask > 0).float()
        return img, mask


# ----------------------------
# 2. 指标计算
# ----------------------------
def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    计算 Dice 系数 (batch 平均)
    pred, target ∈ {0,1}^{N×1×H×W}
    """
    pred_flat   = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = ((2 * inter + eps) / (union + eps)).mean()
    return dice.item()

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    """
    计算 IoU (Jaccard) (batch 平均)
    """
    pred_flat   = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    iou = ((inter + eps) / (union + eps)).mean()
    return iou.item()


# ----------------------------
# 3. 训练 / 验证 / 测试 函数
# ----------------------------
scaler = GradScaler()

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch} [train]"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        # 混合精度上下文
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)

        # 用 scaler 代替 loss.backward() 和 optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """
    在验证或测试集上计算平均 Dice & IoU
    """
    model.eval()
    total_dice = 0.0
    total_iou  = 0.0
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            # 如果输出尺寸小于原掩码，裁剪掩码至相同大小
            if probs.shape[2:] != masks.shape[2:]:
                masks = center_crop(masks, probs)

            preds = (probs > 0.5).float()
            total_dice += dice_coeff(preds, masks)
            total_iou  += iou_score(preds, masks)
    n = len(dataloader)
    return total_dice / n, total_iou / n


# ----------------------------
# 4. 主函数
# ----------------------------
def main():
    # ---- 超参数 & 路径 ----
    TRAIN_IMG_DIR  = "./train/images"
    TRAIN_MASK_DIR = "./train/masks"
    VAL_IMG_DIR    = "./val/images"
    VAL_MASK_DIR   = "./val/masks"
    TEST_IMG_DIR   = "./test/images"
    TEST_MASK_DIR  = "./test/masks"

    
    BATCH_SIZE     = 4
    LR             = 1e-3
    NUM_EPOCHS     = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 数据变换 ----
    INPUT_SIZE  = 256
    OUTPUT_SIZE = 68   # 根据上面 calc_output_size(256) 的结果

    img_transforms = transforms.Compose([
        transforms.ToTensor(),   # 直接 [256×256] → Tensor
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std =[0.229,0.224,0.225]
        ),
    ])

    mask_transforms = transforms.Compose([
        # 已经是256×256了，直接中心裁剪到68×68
        transforms.CenterCrop((OUTPUT_SIZE, OUTPUT_SIZE)),
        transforms.ToTensor(),   # [0,1] 二值掩码
    ])
    # ---- DataLoader ----
    train_ds = ISICDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                           img_transform=img_transforms,
                           mask_transform=mask_transforms)
    val_ds   = ISICDataset(VAL_IMG_DIR,   VAL_MASK_DIR,
                           img_transform=img_transforms,
                           mask_transform=mask_transforms)
    test_ds  = ISICDataset(TEST_IMG_DIR,  TEST_MASK_DIR,
                           img_transform=img_transforms,
                           mask_transform=mask_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # ---- 模型 / 损失 / 优化器 / 调度 ----
    model     = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_dice = 0.0

    # ---- 训练循环 ----
    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model, train_loader,
                                     criterion, optimizer, device, epoch)
        val_dice, val_iou = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}: "
              f"Train Loss {train_loss:.4f} | "
              f"Val Dice {val_dice:.4f} | Val IoU {val_iou:.4f}")

        # 保存最优模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ↳ Saved best model (Dice: {best_val_dice:.4f})")

        scheduler.step()

    # ---- 测试评估 ----
    # 转到test.py
    # print("\nTesting with best model...")
    # model.load_state_dict(torch.load("best_model.pth", map_location=device))
    # test_dice, test_iou = evaluate(model, test_loader, device)
    # print(f"Test Dice: {test_dice:.4f} | Test IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()
