# utils.py

import numpy as np
import torch

def _get_tp_fp_fn_tn(pred, target, threshold=0.5):
    """
    计算二元分割结果的 TP, FP, FN, TN
    这是一个内部辅助函数。

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :return: (tp, fp, fn, tn)
    :四个基本量：真阳性 (True Positives, TP),假阳性 (False Positives, FP),
    真阴性 (True Negatives, TN) 和 假阴性 (False Negatives, FN)
    """
    # 确保输入是 torch tensor, 并与 pred 在同一设备上
    if not torch.is_tensor(pred):
        # 如果 pred 是 numpy, 需要先转为 torch tensor
        pred = torch.from_numpy(pred)
    
    if not torch.is_tensor(target):
        # 使用 as_tensor 避免不必要的数据拷贝，并确保设备和类型一致
        target = torch.as_tensor(target, dtype=torch.bool, device=pred.device)
    else:
        # 确保目标张量是布尔类型
        target = target.bool().to(pred.device) # 确保设备一致

    # 将预测概率图二值化
    pred_binary = (pred > threshold).bool()

    # 计算 TP, FP, FN, TN
    tp = (pred_binary & target).sum()
    fp = (pred_binary & ~target).sum()
    fn = (~pred_binary & target).sum()
    tn = (~pred_binary & ~target).sum()

    return tp, fp, fn, tn

def compute_iou(pred, target, threshold=0.5, epsilon=1e-6):
    """
    计算 Jaccard Index (IoU - Intersection over Union)。
    公式: TP / (TP + FP + FN)

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :param epsilon: 防止除以零的小常数
    :return: IoU 值 (float)
    """
    tp, fp, fn, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    
    # union = tp + fp + fn
    iou = (tp + epsilon) / (tp + fp + fn + epsilon)
    return iou.item()

def compute_dice(pred, target, threshold=0.5, epsilon=1e-6):
    """
    计算 Dice 系数 (F1-Score 的一种形式)。
    公式: 2 * TP / (2 * TP + FP + FN)

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :param epsilon: 防止除以零的小常数
    :return: Dice 值 (float)
    """
    tp, fp, fn, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    
    # total_area = (tp + fp) + (tp + fn) = 2*tp + fp + fn
    dice = (2. * tp + epsilon) / (2. * tp + fp + fn + epsilon)
    return dice.item()

def compute_precision(pred, target, threshold=0.5, epsilon=1e-6):
    """
    计算精确率 (Precision)。
    公式: TP / (TP + FP)

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :param epsilon: 防止除以零的小常数
    :return: Precision 值 (float)
    """
    tp, fp, _, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    precision = (tp + epsilon) / (tp + fp + epsilon)
    return precision.item()

def compute_recall(pred, target, threshold=0.5, epsilon=1e-6):
    """
    计算召回率 (Recall)，也称为灵敏度 (Sensitivity)。
    公式: TP / (TP + FN)

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :param epsilon: 防止除以零的小常数
    :return: Recall 值 (float)
    """
    tp, _, fn, _ = _get_tp_fp_fn_tn(pred, target, threshold)
    recall = (tp + epsilon) / (tp + fn + epsilon)
    return recall.item()

def compute_specificity(pred, target, threshold=0.5, epsilon=1e-6):
    """
    计算特异度 (Specificity)。
    公式: TN / (TN + FP)

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :param epsilon: 防止除以零的小常数
    :return: Specificity 值 (float)
    """
    _, fp, _, tn = _get_tp_fp_fn_tn(pred, target, threshold)
    specificity = (tn + epsilon) / (tn + fp + epsilon)
    return specificity.item()

def compute_accuracy(pred, target, threshold=0.5, epsilon=1e-6):
    """
    计算准确率 (Accuracy)。
    公式: (TP + TN) / (TP + TN + FP + FN)

    :param pred: 预测结果 (概率图)，numpy 数组或 torch tensor
    :param target: 真实标签 (二值图), numpy 数组或 torch tensor
    :param threshold: 用于将预测概率图二值化的阈值
    :param epsilon: 防止除以零的小常数
    :return: Accuracy 值 (float)
    """
    tp, fp, fn, tn = _get_tp_fp_fn_tn(pred, target, threshold)
    accuracy = (tp + tn + epsilon) / (tp + tn + fp + fn + epsilon)
    return accuracy.item()