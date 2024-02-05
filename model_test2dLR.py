import random

import torch
import json
import os
import numpy as np
from model2 import *
from torch.nn import functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch.utils.data import TensorDataset, random_split

joint_relations = {
    0: {"name": "PELVIS", "parent": None},
    1: {"name": "SPINE_NAVAL", "parent": 0},
    2: {"name": "SPINE_CHEST", "parent": 1},
    3: {"name": "NECK", "parent": 2},
    4: {"name": "CLAVICLE_LEFT", "parent": 2},
    5: {"name": "SHOULDER_LEFT", "parent": 4},
    6: {"name": "ELBOW_LEFT", "parent": 5},
    7: {"name": "WRIST_LEFT", "parent": 6},
    8: {"name": "HAND_LEFT", "parent": 7},
    9: {"name": "HANDTIP_LEFT", "parent": 8},
    10: {"name": "THUMB_LEFT", "parent": 7},
    11: {"name": "CLAVICLE_RIGHT", "parent": 2},
    12: {"name": "SHOULDER_RIGHT", "parent": 11},
    13: {"name": "ELBOW_RIGHT", "parent": 12},
    14: {"name": "WRIST_RIGHT", "parent": 13},
    15: {"name": "HAND_RIGHT", "parent": 14},
    16: {"name": "HANDTIP_RIGHT", "parent": 15},
    17: {"name": "THUMB_RIGHT", "parent": 14},
    18: {"name": "HIP_LEFT", "parent": 0},
    19: {"name": "KNEE_LEFT", "parent": 18},
    20: {"name": "ANKLE_LEFT", "parent": 19},
    21: {"name": "FOOT_LEFT", "parent": 20},
    22: {"name": "HIP_RIGHT", "parent": 0},
    23: {"name": "KNEE_RIGHT", "parent": 22},
    24: {"name": "ANKLE_RIGHT", "parent": 23},
    25: {"name": "FOOT_RIGHT", "parent": 24},
    26: {"name": "HEAD", "parent": 3},
    27: {"name": "NOSE", "parent": 26},
    28: {"name": "EYE_LEFT", "parent": 26},
    29: {"name": "EAR_LEFT", "parent": 26},
    30: {"name": "EYE_RIGHT", "parent": 26},
    31: {"name": "EAR_RIGHT", "parent": 26},
}

def create_dataset(inputsxyL, inputsxyR, inputsyz, targets, sequence_size=1):
    dataset = []

    for i in range(len(inputsxyL) - sequence_size + 1):
        input_sequence = (inputsxyL[i:i + sequence_size], inputsxyR[i:i + sequence_size], inputsyz[i:i + sequence_size])
        target_sequence = targets[i+sequence_size-1]
        dataset.append((input_sequence, target_sequence))

    return dataset

def plot_joint_lines(ax, output, joint_relations):
    # 関節同士を線で結ぶ
    lines = []
    for joint_index, joint_info in joint_relations.items():
        parent_index = joint_info["parent"]
        if parent_index is not None:
            line = [(output[parent_index * 3], output[parent_index * 3 + 1], output[parent_index * 3 + 2]),
                    (output[joint_index * 3], output[joint_index * 3 + 1], output[joint_index * 3 + 2])]
            lines.append(line)

    # 線のコレクションを作成して追加
    lc = Line3DCollection(lines, color='black', linestyle='solid', alpha=0.5)
    ax.add_collection3d(lc)

num_flames = 1694
batch_size = 1  # テストバッチサイズ
num_points = 550
num_labels = 96

# テストデータの読み込み
test_inputsxyL = []
test_inputsxyR = []
test_inputsyz = []
test_targets = []

test_dir = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/ScannedTwoRadars_tilt4"
for i in range(num_flames+1):
    file_name = f"{i}.json"
    file_path = os.path.join(test_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            # Check if all values in PointCloud are NaN
            if np.isnan(data["PointCloud"]).all():
                # Skip this frame
                continue

            batch_listL = []
            batch_listR = []
            for x in range(data["NumPoints"]):
                if data["PointCloud"][0][x] < 0:
                    batch_listL.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][4][x]])
                if data["PointCloud"][0][x] >= 0:
                    batch_listR.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][4][x]])

            # Check if NumPoints is less than num_points
            if len(batch_listL) < num_points:
                # Fill the remaining points with NaN
                batch_listL += [[0, 0, 0]] * (num_points - len(batch_listL))
            if len(batch_listR) < num_points:
                # Fill the remaining points with NaN
                batch_listR += [[0, 0, 0]] * (num_points - len(batch_listR))

            test_inputsxyL.append(batch_listL)
            test_inputsxyR.append(batch_listR)

            test_targets.append(data["KeyPoint"][0])

for i in range(num_flames+1):
    file_name = f"{i}.json"
    file_path = os.path.join(test_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            # Check if all values in PointCloud are NaN
            if np.isnan(data["PointCloud"]).all():
                # Skip this frame
                continue

            batch_list = []
            for x in range(data["NumPoints"]):
                batch_list.append([data["PointCloud"][1][x], data["PointCloud"][2][x], data["PointCloud"][4][x]])

            # Check if NumPoints is less than num_points
            if len(batch_list) < num_points:
                # Fill the remaining points with NaN
                batch_list += [[0, 0, 0]] * (num_points - len(batch_list))

            test_inputsyz.append(batch_list)

print(len(test_inputsxyL))
print(len(test_inputsxyL[0]))
print(len(test_inputsxyL[0][0]))
test_inputsxyL = np.array(test_inputsxyL)
print(len(test_inputsxyR))
print(len(test_inputsxyR[0]))
print(len(test_inputsxyR[0][0]))
test_inputsxyR = np.array(test_inputsxyR)
print(len(test_inputsyz))
print(len(test_inputsyz[0]))
print(len(test_inputsyz[0][0]))
test_inputsyz = np.array(test_inputsyz)
test_targets = np.array(test_targets)

min_values = test_targets.min(axis=0)
max_values = test_targets.max(axis=0)
#print(min_values.mean())
#print(max_values.mean())
test_targets = (test_targets - min_values) / (max_values - min_values)

test_inputsxyL = torch.from_numpy(test_inputsxyL)
test_inputsxyR = torch.from_numpy(test_inputsxyR)
test_inputsyz = torch.from_numpy(test_inputsyz)
test_targets = torch.from_numpy(test_targets)
test_inputsxyL = test_inputsxyL.to(torch.float32)
test_inputsxyR = test_inputsxyR.to(torch.float32)
test_inputsyz = test_inputsyz.to(torch.float32)
test_targets = test_targets.to(torch.float32)

data = TensorDataset(test_inputsxyL, test_inputsxyR, test_inputsyz, test_targets)
dataset = create_dataset(data.tensors[0], data.tensors[1], data.tensors[2], data.tensors[3])
num_iterations_per_epoch = len(dataset) // batch_size

# 学習済みモデルの読み込み
model = PointNetmmPoseCNN2(num_points, num_labels).cuda()
model.load_state_dict(torch.load("C:/Users/kodam/PycharmProjects/mmModels/Models/TwoRadars_tilt_mmPose/saved_model_23.pth"))


mae_values = []
mae_values_ind = []

fig = plt.figure(figsize=(18, 12))
fig.suptitle('KeyPoints Plot of Predicted and True Values')

# 3D グラフの設定
ax_output = fig.add_subplot(121, projection='3d')
ax_targets = fig.add_subplot(122, projection='3d')

for iteration in range(num_iterations_per_epoch):
    indices = random.sample(range(len(dataset)), 1)
    data = [dataset[i] for i in indices]

    #ts_inputs = torch.Tensor(ts_item[0]).unsqueeze(0).squeeze(dim=0).cuda()
    #ts_targets = torch.Tensor(ts_item[1]).unsqueeze(0).cuda()
    ts_inputsxyL = torch.stack([torch.Tensor(item[0][0]) for item in data]).squeeze(dim=0).cuda()
    ts_inputsxyR = torch.stack([torch.Tensor(item[0][1]) for item in data]).squeeze(dim=0).cuda()
    ts_inputsyz = torch.stack([torch.Tensor(item[0][2]) for item in data]).squeeze(dim=0).cuda()
    #print(ts_inputs)
    middle_index = len(data) // 2
    ts_targets = torch.Tensor(data[middle_index][1]).unsqueeze(0).cuda()

    #model.eval()
    ts_output = model(ts_inputsxyL, ts_inputsxyR, ts_inputsyz)

    # 逆正規化された値でMAEを計算
    restored_output = (ts_output.squeeze().cpu().detach().numpy() * (max_values - min_values)) + min_values
    restored_targets_batch = (ts_targets.squeeze().cpu().detach().numpy() * (max_values - min_values)) + min_values
    mae = np.abs(restored_output - restored_targets_batch)
    #mae = ((restored_output - restored_targets_batch)**2)

    # 逆正規化なし
    #mae = np.abs(ts_output.squeeze().cpu().detach().numpy() - ts_targets.squeeze().cpu().detach().numpy())
    #mae = ((ts_output.squeeze().cpu().detach().numpy() - ts_targets.squeeze().cpu().detach().numpy())**2)

    # 各座標のMAEを計算
    pointwise_mae = mae

    mae_values_ind.append(pointwise_mae)

    mae_values.append(mae.mean())

    # Predicted Values
    ax_output.scatter(restored_output[0::3],
                      restored_output[1::3],
                      restored_output[2::3],
                      label=f'flame {iteration + 1}', alpha=0.5)

    # True Values
    ax_targets.scatter(restored_targets_batch[0::3],
                       restored_targets_batch[1::3],
                       restored_targets_batch[2::3],
                       label=f'flame {iteration + 1}', alpha=0.5, marker='x')

    # 関節同士を線で結ぶ関数を呼び出し
    plot_joint_lines(ax_output, restored_output, joint_relations)
    plot_joint_lines(ax_targets, restored_targets_batch, joint_relations)

    # ラベルの設定
    ax_output.set_xlabel('X')
    ax_output.set_ylabel('Y')
    ax_output.set_zlabel('Z')
    ax_output.set_title('PointNetPose Output')
    ax_output.legend()

    ax_targets.set_xlabel('X')
    ax_targets.set_ylabel('Y')
    ax_targets.set_zlabel('Z')
    ax_targets.set_title('Azure Kinect Output')
    ax_targets.legend()

    min = np.min(min_values)
    max = np.max(max_values)
    ax_output.set_xlim([-800, 600])
    ax_output.set_ylim([1000, 4500])
    ax_output.set_zlim([0, 2000])

    ax_targets.set_xlim([-800, 600])
    ax_targets.set_ylim([1000, 4500])
    ax_targets.set_zlim([0, 2000])

    # 表示
    plt.pause(0.5)

    # グラフをクリア
    ax_output.clear()
    ax_targets.clear()

    #plt.plot(mae_values, label='MAE')
    #plt.xlabel('flame')
    #plt.ylabel('MAE Value')
    #plt.title('Real-time MAE Plot')
    #plt.legend()
    #plt.pause(0.1)  # Adjust the pause duration as needed
    #plt.clf()  # Clear the plot for the next iteration

    print('MAE on test data:', mae.mean())
    #print('Predictions:', ts_output)
    #print('True values:', ts_targets)

# 各イテレーションのMAEの平均を計算
average_pointwise_mae = np.mean(mae_values_ind, axis=0)
print(average_pointwise_mae)

print('==============================================')

print('Overall MAE Average: ', np.mean(mae_values))

print('Average Pointwise MAE: ')
for coord_index in range(len(average_pointwise_mae)):
    coord_mae = average_pointwise_mae[coord_index]
    print(f'Coordinate {coord_index}: {coord_mae}')

x = average_pointwise_mae[::3]
y = average_pointwise_mae[1::3]
z = average_pointwise_mae[2::3]
plt.figure(figsize=(10, 6))
plt.plot(x, label='Azimuth')
plt.plot(y, label='Depth')
plt.plot(z, label='Elevation')
plt.xlabel('Joint Index')
plt.ylabel('Error(mm)')
plt.title('Mean Absolute Error')
plt.legend()
plt.show()

# Final plot
plt.plot(mae_values, label='MAE')
plt.xlabel('Frame')
plt.ylabel('MAE Value')
plt.title('Final MAE Plot')
plt.legend()
plt.show()
