# -*- coding：UTF-8 -*-
'''
@Project : DIP-HyperKite-main
@File ：tensor_rotate.py
@Author : Zerbo
@Date : 2024/2/28 10:16
'''

#旋转lrhsi一个小角度制造配准误差
import torch
import math
from torch.nn import functional as F

import cv2
import numpy as np

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w//2, h//2)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, border_Value=(0, 0, 0))

    cos = np.abs(rotate_matrix[0,0])
    sin = np.abs(rotate_matrix[0,0])
    new_w = int(h*sin+w*cos)
    new_h = int(h*cos+w*sin)
    rotate_matrix[0, 2] += (new_w/2)-center[0]
    rotate_matrix[1, 2] += (new_h/2) - center[1]

    rotated_image = cv2.warpAffine(rotated_image, rotate_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, border_Value=(0, 0, 0))

    return rotated_image


def rotate_tensor(image, angle):
    # 获取图像尺寸
    H, W = image.size(2), image.size(3)

    # 确定旋转中心
    center_x, center_y = W / 2, H / 2

    # 计算旋转矩阵
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta], [sin_theta, cos_theta]], device=image.device)

    # 生成网格坐标
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))

    # 将坐标转化为相对旋转中心的坐标
    relative_x = grid_x - center_x
    relative_y = grid_y - center_y

    # 进行坐标变换
    rotated_x = relative_x * cos_theta - relative_y * sin_theta + center_x
    rotated_y = relative_x * sin_theta + relative_y * cos_theta + center_y

    # 进行双线性插值计算
    rotated_image = torch.nn.functional.grid_sample(image, torch.stack([rotated_x, rotated_y], dim=-1).unsqueeze(0))

    return rotated_image.squeeze()