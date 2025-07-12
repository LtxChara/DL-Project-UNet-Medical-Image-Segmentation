#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np

def load_best_models():
    data = {
        'model': [
            'UNet_with_Augmentation_fold1_best.pth', 'UNet_with_Augmentation_fold2_best.pth',
            'UNet_with_Augmentation_fold3_best.pth', 'UNet_with_Augmentation_fold4_best.pth',
            'UNet_with_Augmentation_fold5_best.pth',
            'UNet_without_Augmentation_fold1_best.pth', 'UNet_without_Augmentation_fold2_best.pth',
            'UNet_without_Augmentation_fold3_best.pth', 'UNet_without_Augmentation_fold4_best.pth',
            'UNet_without_Augmentation_fold5_best.pth',
            'UNet++_with_Augmentation_fold1_best.pth', 'UNet++_with_Augmentation_fold2_best.pth',
            'UNet++_with_Augmentation_fold3_best.pth', 'UNet++_with_Augmentation_fold4_best.pth',
            'UNet++_with_Augmentation_fold5_best.pth',
            'UNet++_without_Augmentation_fold1_best.pth', 'UNet++_without_Augmentation_fold2_best.pth',
            'UNet++_without_Augmentation_fold3_best.pth', 'UNet++_without_Augmentation_fold4_best.pth',
            'UNet++_without_Augmentation_fold5_best.pth',
            'HMTUNet_with_Augmentation_fold1_best.pth','HMTUNet_with_Augmentation_fold2_best.pth',
            'HMTUNet_with_Augmentation_fold3_best.pth','HMTUNet_with_Augmentation_fold4_best.pth',
            'HMTUNet_with_Augmentation_fold5_best.pth',
            'HMTUNet_without_Augmentation_fold1_best.pth','HMTUNet_without_Augmentation_fold2_best.pth',
            'HMTUNet_without_Augmentation_fold3_best.pth','HMTUNet_without_Augmentation_fold4_best.pth',
            'HMTUNet_without_Augmentation_fold5_best.pth'

        ],
        'iou': [0.5157, 0.5138, 0.5637, 0.5628, 0.5777,
                0.5526, 0.4364, 0.4315, 0.4724, 0.3955,
                0.6509, 0.5825, 0.5579, 0.6551, 0.5975,
                0.3905, 0.3261, 0.2683, 0.5534, 0.4106,
                0.6158, 0.5854, 0.6014, 0.5774, 0.5461,
                0.5559, 0.5319, 0.5092, .4376, 0.5313],
        'dice': [0.6031, 0.6011, 0.6531, 0.6495, 0.6664,
                 0.6524, 0.5198, 0.5127, 0.5636, 0.4738,
                 0.7490, 0.6689, 0.6409, 0.7488, 0.6861,
                 0.4679, 0.3903, 0.3358, 0.6510, 0.4879,
                 0.7087, 0.6725, 0.6895, 0.6598, 0.6483,
                 0.6604, 0.6272, 0.6116, 0.5337, 0.6289],
        'precision': [
            0.959296077, 0.976565312, 0.95441493, 0.950320367, 0.941005484,
            0.908932998, 0.974078747, 0.973428161, 0.963259819, 0.973680799,
            0.905988293, 0.953945986, 0.967961464, 0.934878758, 0.933698996,
            0.975722732, 0.97586393, 0.990390376, 0.928648651, 0.974776745,
            0.935758285, 0.937130066, 0.936817183, 0.929676971, 0.89649295, 
            0.901958453, 0.921419869, 0.907421922, 0.909900683, 0.927298807
        ],
        'recall': [0.5454, 0.5285, 0.5991, 0.5974, 0.6226,
                   0.6059, 0.4478, 0.4473, 0.4902, 0.4059,
                   0.7240, 0.6164, 0.5830, 0.7035, 0.6493,
                   0.4018, 0.3393, 0.2726, 0.5939, 0.4224,
                   0.6576, 0.6235, 0.6371, 0.6234, 0.6125,
                   0.6056, 0.5663, 0.5522, 0.4794,0.5710],
        'accuracy': [0.8427, 0.8483, 0.8566, 0.8574, 0.8620,
                     0.8627, 0.8291, 0.8362, 0.8459, 0.8313,
                     0.8861, 0.8640, 0.8621, 0.8860, 0.8667,
                     0.8224, 0.8207, 0.8080, 0.8592, 0.8319,
                     0.8792, 0.8707, 0.8769, 0.8671, 0.8513,
                     0.8714, 0.8598, 0.8534, 0.8296, 0.8550],
        'specificity': [
            0.996935695, 0.997606129, 0.994954144, 0.99450864, 0.993338905,
            0.990198136, 0.996689097, 0.995563008, 0.99404052, 0.996503222,
            0.987871804, 0.993510498, 0.994910299, 0.990041812, 0.991943442,
            0.997301427, 0.99621719, 0.998640863, 0.990951199, 0.996142695,
            0.989131581, 0.988228146, 0.988938579, 0.989002331, 0.989802005,
            0.981599923, 0.989829895, 0.986520175, 0.990441566, 0.990454695
        ]

    }
    df = pd.DataFrame(data)
    df['Config'] = df['model'].apply(
        lambda n: ('HMTUNet' if n.startswith('HMTUNet') else  ('UNet++' if n.startswith('UNet++') else 'UNet')) +
                  (' w/ Aug' if 'with_Augmentation' in n else ' w/o Aug')
    )
    best = df.loc[df.groupby('Config')['iou'].idxmax()].reset_index(drop=True)
    return best

def plot_radar(df):
    # 准备数据
    metrics = ['iou', 'dice', 'precision', 'recall', 'accuracy', 'specificity']
    metrics = ['iou', 'dice', 'recall', 'accuracy']
    labels = [m.upper() for m in metrics]
    num_vars = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    for i, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row['Config'], linewidth=2)
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title('Radar Chart of Best Models', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def plot_parallel(df):
    # 平行坐标图需要配置为字符串标签列
    df_pc = df[['Config', 'iou', 'dice', 'recall', 'accuracy']].copy()
    plt.figure(figsize=(8,5))
    parallel_coordinates(df_pc, 'Config', colormap=plt.cm.Set2)
    plt.title('Parallel Coordinates of Best Models')
    plt.ylabel('Metric Value')
    plt.legend(title='Config', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_bar(df, colors):
    metrics = ['iou', 'dice', 'precision', 'recall', 'accuracy', 'specificity']
    labels = [m.upper() for m in metrics]
    x = np.arange(len(metrics))
    total_width = 0.8
    n = len(df)
    width = total_width / n

    fig, ax = plt.subplots(figsize=(8,5))
    for i, row in df.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i*width, vals, width=width, label=row['Config'], color=colors[i])

    ax.set_xticks(x + total_width/2 - width/2)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Metric Value')
    ax.set_title('Bar Chart of Best Models', loc='left')
    ax.legend(title='Config', loc='upper right')
    plt.tight_layout()
    plt.show()

def main():
    best = load_best_models()

    plot_radar(best)
    plot_parallel(best)

    cmap = plt.cm.Set2
    colors = [cmap(i) for i in range(len(best))]
    plot_bar(best, colors)
    

if __name__ == '__main__':
    main()
