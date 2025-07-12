# test.py

import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
# 指标函数

from utils import (
    compute_iou, compute_dice, compute_precision,
    compute_recall, compute_specificity, compute_accuracy
)

# ==== 模型结构导入 ====
# from unet_base import UNet  # UNet 类定义文件
from model import UNet, UNetPlusPlus  # UNet 类定义文件
from hmt_unet import HMTUNet

# ==== 自定义 Dataset ====
class TestDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # 灰度图

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# def center_crop(tensor, target_size):
#     _, _, h, w = tensor.shape
#     th, tw = target_size
#     i = (h - th) // 2
#     j = (w - tw) // 2
#     return tensor[:, :, i:i+th, j:j+tw]

def evaluate(model, dataloader, device):
    model.eval()
    metrics_accum = {
        'iou': [], 'dice': [], 'prec': [],
        'rec': [], 'acc': [], 'spec': []
    }

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluating', leave=False):
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            preds   = torch.sigmoid(outputs).cpu().numpy()
            masks_np= masks.cpu().numpy()

            for pred, target in zip(preds, masks_np):
                pred = np.squeeze(pred)
                target = np.squeeze(target) / 255.0

                metrics_accum['iou'].append(  compute_iou(pred, target) )
                metrics_accum['dice'].append( compute_dice(pred, target) )
                metrics_accum['prec'].append( compute_precision(pred, target) )
                metrics_accum['rec'].append(  compute_recall(pred, target) )
                metrics_accum['acc'].append(  compute_accuracy(pred, target) )
                metrics_accum['spec'].append( compute_specificity(pred, target) )

    # **一定要在这里返回字典**，且缩进与函数体对齐
    mean_metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
    return mean_metrics

def visualize_random_samples(model, dataset, device, num_samples=5, threshold=0.5):
    """
    随机抽取 num_samples 个样本，绘制原图、真实掩码与预测掩码并排对比。

    Args:
        model:           已加载权重并处于 eval 模式的分割模型
        dataset:         TestDataset 对象
        device:          torch.device('cuda' or 'cpu')
        num_samples:     要可视化的样本数量
        threshold:       预测概率转二值掩码的阈值
    """
    model.eval()
    # 随机抽样
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    # 用于把 tensor 图像 -> numpy H×W×C
    def tensor_to_image(tensor):
        img = tensor.cpu().numpy().transpose(1, 2, 0)  # C×H×W -> H×W×C
        img = np.clip(img, 0, 1)  # 假设 ToTensor 后范围在 [0,1]
        return img

    # 遍历每个样本并绘图
    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, 0)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            # 取数据
            image, mask = dataset[idx]
            x = image.unsqueeze(0).to(device)  # 1×C×H×W

            # 推理
            pred_logits = model(x)
            pred_prob = torch.sigmoid(pred_logits)[0, 0]  # H×W
            pred_mask = (pred_prob.cpu().numpy() > threshold).astype(np.uint8)

            # 真值掩码
            gt_mask = mask.squeeze(0).cpu().numpy()        # H×W, 0/1 or 0/255

            # 绘制
            ax_orig, ax_gt, ax_pred = axes[row]
            ax_orig.imshow(tensor_to_image(image))
            ax_orig.set_title("Original Image")
            ax_orig.axis("off")

            ax_gt.imshow(gt_mask, cmap="gray")
            ax_gt.set_title("Ground Truth")
            ax_gt.axis("off")

            ax_pred.imshow(pred_mask, cmap="gray")
            ax_pred.set_title("Predicted Mask")
            ax_pred.axis("off")

    plt.tight_layout()
    plt.show()
  
# ==== 主程序入口 ====
if __name__ == '__main__':
    # 五折模型路径
    model_paths = [
        f'Weights/HMTUNet without aug/HMTUNet_without_Augmentation_fold{i}_best.pth'
        for i in range(1,6)

    ]

    # 选择模型结构（UNet, UNetPlusPlus 或 HMTUNet）
    ModelClass = HMTUNet  

    # 测试集路径
    test_image_dir = 'Datas/processed/test/images'
    test_mask_dir  = 'Datas/processed/test/masks'

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])

    # 加载数据
    test_dataset = TestDataset(test_image_dir, test_mask_dir, transform)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 存储每个模型的评估结果
    records = []

    for path in model_paths:
        print(f'\n>> Evaluating model: {path}')
        model = ModelClass(n_channels=3, n_classes=1).to(device)
        model.load_state_dict(torch.load(path, map_location=device))

        metrics = evaluate(model, test_loader, device)
        metrics['model'] = os.path.basename(path)
        records.append(metrics)

        print("Result:", metrics)

    # 汇总结果为 DataFrame
    df = pd.DataFrame(records)
    df = df[['model','iou','dice','prec','rec','acc','spec']]

    print("\n=== All folds results ===")
    print(df.to_string(index=False))

    # 选出平均 Dice 最高的模型
    best = df.loc[df['dice'].idxmax()]
    print(f"\n*** Best model: {best['model']} (Dice={best['dice']:.4f}) ***")

    # 保存到 CSV：
    df.to_csv('evaluation_summary.csv', index=False)
    print("Saved summary to evaluation_summary.csv")

    #最优模型可视化
    print(f"\nVisualizing samples from best model: {best['model']}")
    best_model = ModelClass(n_channels=3, n_classes=1).to(device)
    best_model.load_state_dict(torch.load(os.path.join('Weights/UNET++ with aug', best['model']), map_location=device))
    visualize_random_samples(best_model, test_dataset, device, num_samples=4, threshold=0.5)
