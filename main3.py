from torch.utils.data import TensorDataset, random_split

from model3 import *

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import random

num_flames = 10000
im_size = 100
num_channels = 3
num_labels = 96
epochs = 100

inputs_xy = []
inputs_xz = []
targets = []

# データセット構築

inputs_xy = np.array(inputs_xy)
inputs_xz = np.array(inputs_xz)
targets = np.array(targets)

# キーポイント座標の正規化
min_values = targets.min(axis=0)
max_values = targets.max(axis=0)
targets = (targets - min_values) / (max_values - min_values)

inputs_xy = torch.from_numpy(inputs_xy)
inputs_xz = torch.from_numpy(inputs_xz)
targets = torch.from_numpy(targets)
inputs_xy = inputs_xy.to(torch.float32)
inputs_xz = inputs_xz.to(torch.float32)
targets = targets.to(torch.float32)

def main(inputs_xy, inputs_xz, targets, num_epochs = epochs):
    network = mmPose(im_size, num_channels, num_labels).cuda()

    data = TensorDataset(inputs_xy, inputs_xz, targets)
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size

    train_dataset, val_dataset = random_split(data, [train_size, val_size])

    