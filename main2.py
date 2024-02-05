from __future__ import print_function
import argparse
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model2 import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F

import json
import os
import numpy as np
import matplotlib.pyplot as plt

batch_size = 11023  # フレームの数
num_points = 550  # 点群数
num_labels = 96  # 出力ラベル数
epochs = 100
windowsize = 1

inputs = []
targets = []

dir = "C:/Users/kodam/PycharmProjects/mmModels/CreatedData2"

for i in range(batch_size):
    file_name = f"{i}.json"
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            batch_list = []
            for x in range(data["NumPoints"]):
                batch_list.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][2][x]])
            batch_list += [[0, 0, 0]] * (num_points - len(batch_list))
            inputs.append(batch_list)

            targets.append(data["KeyPoint"][0])

inputs = np.array(inputs)
targets = np.array(targets)

# キーポイント座標の正規化
min_values = targets.min(axis=0)
max_values = targets.max(axis=0)
targets = (targets - min_values) / (max_values - min_values)

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
inputs = inputs.to(torch.float32)
targets = targets.to(torch.float32)


def main(input_data, target):
    pointnet = PointNetCls(num_points, True)

    '''
    new_param = pointnet.state_dict()
    new_param['main.0.main.6.bias'] = torch.eye(3, 3).view(-1)
    new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
    pointnet.load_state_dict(new_param)
    '''

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(pointnet.parameters(), lr=0.001)
    pointnet.cuda()

    loss_list = []
    accuracy_list = []

    for iteration, batch in enumerate(input_data):
        torch.autograd.set_detect_anomaly(True)

        #batch = batch.transpose(1, 2)
        batch, target = batch.cuda(), target.cuda()
        optimizer.zero_grad()
        pointnet = pointnet.train()
        # print(batch.view(-1, 3))
        pred, trans, trans_feat = pointnet(batch)  # 予測された骨格キーポイント座標

        # print(target[iteration])
        error = criterion(pred, torch.unsqueeze(target[iteration], 0))  # 正解の座標と予測の座標とのMAEが小さくなるようにする損失関数
        # print(error)
        error.backward()

        optimizer.step()

        if iteration % 10 == 0:
            with torch.no_grad():
                loss_list.append(error.item())

            print('Epoch : {}   Loss : {}'.format(iteration, error.item()))

    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('MSELoss')
    plt.show()


main(inputs, targets)