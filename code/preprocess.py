import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

def process_isic_data(root_path, output_path, img_size=(256, 256)):
    """
    对ISIC 2017数据集进行预处理，包括统一尺寸、二值化掩码，并保存到新的目录结构中。

    参数:
    - root_path (str): 包含原始ISIC数据集文件夹的根目录。
    - output_path (str): 处理后数据的保存目录。
    - img_size (tuple): 目标图像和掩码的尺寸, 默认为(256, 256)。
    """
    print(f"开始预处理ISIC 2017数据集，目标尺寸: {img_size}")

    # 定义原始数据和目标输出的目录结构
    # 元组格式：(源图像文件夹名, 源掩码文件夹名, 目标保存文件夹名)
    data_map = [
        ("ISIC-2017_Training_Data", "ISIC-2017_Training_Part1_GroundTruth", "train"),
        ("ISIC-2017_Validation_Data", "ISIC-2017_Validation_Part1_GroundTruth", "val"),
        ("ISIC-2017_Test_v2_Data", "ISIC-2017_Test_v2_Part1_GroundTruth", "test")
    ]

    # 遍历训练集、验证集和测试集
    for img_folder_name, mask_folder_name, split_name in data_map:
        print(f"\n===== 正在处理 {split_name} 数据集 =====")

        # 拼接出完整的源文件夹路径
        img_folder = os.path.join(root_path, img_folder_name)
        mask_folder = os.path.join(root_path, mask_folder_name)

        # 【关键改进】在处理前，检查源文件夹是否存在
        if not os.path.isdir(img_folder):
            print(f"错误: 找不到图像文件夹: '{img_folder}'")
            print("请检查您的 RAW_DATA_ROOT 路径和文件夹结构是否正确。已跳过此数据集。")
            continue
        if not os.path.isdir(mask_folder):
            print(f"错误: 找不到掩码文件夹: '{mask_folder}'")
            print("请检查您的 RAW_DATA_ROOT 路径和文件夹结构是否正确。已跳过此数据集。")
            continue
        
        # 获取所有原始图像的路径
        image_paths = glob.glob(os.path.join(img_folder, "*.jpg"))
        
        # 检查是否找到了任何图像
        if not image_paths:
            print(f"警告: 在文件夹 '{img_folder}' 中没有找到任何 .jpg 图像。请检查路径或文件扩展名。")
            continue

        # 创建处理后数据的保存目录
        output_img_folder = os.path.join(output_path, split_name, "images")
        output_mask_folder = os.path.join(output_path, split_name, "masks")
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_mask_folder, exist_ok=True)
        
        # 使用tqdm创建进度条
        for img_path in tqdm(image_paths, desc=f"处理 {split_name} 图像"):
            try:
                # 1. 构建对应的掩码路径
                base_name = os.path.basename(img_path).replace(".jpg", "")
                mask_name = f"{base_name}_segmentation.png"
                mask_path = os.path.join(mask_folder, mask_name)

                # 2. 读取图像和掩码
                image = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"警告: 无法读取图像文件: {img_path}")
                    continue
                if mask is None:
                    print(f"警告: 找不到或无法读取对应的掩码文件: {mask_path}")
                    continue

                # 3. 统一尺寸
                image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

                # 4. 掩码二值化
                _, mask_binary = cv2.threshold(mask_resized, 128, 1, cv2.THRESH_BINARY)

                # 5. 构建输出文件路径并保存
                output_img_path = os.path.join(output_img_folder, os.path.basename(img_path))
                output_mask_path = os.path.join(output_mask_folder, mask_name)

                cv2.imwrite(output_img_path, image_resized)
                cv2.imwrite(output_mask_path, mask_binary.astype(np.uint8) * 255)

            except Exception as e:
                print(f"处理文件 {img_path} 时发生未知错误: {e}")

    print("\n数据预处理完成！所有成功处理的文件已保存至: ", os.path.abspath(output_path))

if __name__ == '__main__':

    RAW_DATA_ROOT = "./data/" 
    
    # 定义处理后数据的保存目录
    PROCESSED_DATA_PATH = "./data/processed"
    
    # 定义模型输入的目标尺寸
    TARGET_SIZE = (256, 256)
    
    # --- 执行预处理 ---
    process_isic_data(root_path=RAW_DATA_ROOT, 
                      output_path=PROCESSED_DATA_PATH, 
                      img_size=TARGET_SIZE)
