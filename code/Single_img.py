import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from skimage import measure
from model import UNet, UNetPlusPlus  # 或者 UNetPlusPlus
from hmt_unet import HMTUNet

# ===== 配置区 =====
# 模型权重路径
Unet_without = 'Weights/UNET without aug/UNet_without_Augmentation_fold1_best.pth'
Unet_with = 'Weights/UNET with aug/UNet_with_Augmentation_fold1_best.pth' # 对测试图片最优、得分最优为fold5
UnetPLUS_without = 'Weights/UNET++ without aug/UNet++_without_Augmentation_fold1_best.pth' # 对测试图片最优、得分最优为fold4
UnetPLUS_with = 'Weights/UNET++ with aug/UNet++_with_Augmentation_fold2_best.pth'
HMTUNet_without = 'Weights/HMTUNet without aug/HMTUNet_without_Augmentation_fold1_best.pth' #都是fold1
HMTUNet_with = 'Weights/HMTUNet with aug/HMTUNet_with_Augmentation_fold1_best.pth'

model_path = UnetPLUS_without

# 测试图像和掩码路径 ISIC_0012199 ISIC_0012330
image_path = 'Datas/processed/test/images/ISIC_0012330.jpg'
mask_path  = 'Datas/processed/test/masks/ISIC_0012330_segmentation.png'

# 二值化阈值
threshold  = 0.5
# 整体标题和图例
figure_title = 'Single Model, Single Sample Visualization'
legend_labels = ['Ground Truth (Green)', 'Prediction (Red)']

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 预处理
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# ===== 加载模型 =====
def load_model(path):
    model = HMTUNet(n_channels=3, n_classes=1).to(device) # 切换模型
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

# ===== 可视化函数 =====
def visualize_single():
    # 加载
    model = load_model(model_path)
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    # 推理
    img_t = transform(img).to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_t)
        prob   = torch.sigmoid(logits)[0,0].cpu().numpy()
    pred_bin = (prob > threshold).astype(np.uint8)

    # 实际掩码
    gt_bin = (np.array(mask.resize((256,256))) > 0).astype(np.uint8)

    # 查找轮廓
    contours_gt   = measure.find_contours(gt_bin,   0.5)
    contours_pred = measure.find_contours(pred_bin, 0.5)

    # 绘图
    plt.figure(figsize=(6,6))
    plt.imshow(np.clip(np.array(img.resize((256,256))) / 255.0, 0, 1))
    for c in contours_gt:
        plt.plot(c[:,1], c[:,0], linewidth=2, color='g')
    for c in contours_pred:
        plt.plot(c[:,1], c[:,0], linewidth=2, color='r')

    # plt.title(figure_title)
    # 添加图例
    handles = [
        plt.Line2D([0], [0], color='g', lw=2),
        plt.Line2D([0], [0], color='r', lw=2)
    ]
    plt.legend(handles, legend_labels, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_single()
