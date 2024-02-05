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

joint_relations32 = {
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

joint_relations22 = {
    0: {"name": "PELVIS", "parent": None},
    1: {"name": "SPINE_NAVAL", "parent": 0},
    2: {"name": "SPINE_CHEST", "parent": 1},
    3: {"name": "NECK", "parent": 2},
    4: {"name": "SHOULDER_LEFT", "parent": 3},
    5: {"name": "ELBOW_LEFT", "parent": 4},
    6: {"name": "SHOULDER_RIGHT", "parent": 3},
    7: {"name": "ELBOW_RIGHT", "parent": 6},
    8: {"name": "HIP_LEFT", "parent": 0},
    9: {"name": "KNEE_LEFT", "parent": 8},
    10: {"name": "ANKLE_LEFT", "parent": 9},
    11: {"name": "FOOT_LEFT", "parent": 10},
    12: {"name": "HIP_RIGHT", "parent": 0},
    13: {"name": "KNEE_RIGHT", "parent": 12},
    14: {"name": "ANKLE_RIGHT", "parent": 13},
    15: {"name": "FOOT_RIGHT", "parent": 14},
    16: {"name": "HEAD", "parent": 3},
    17: {"name": "NOSE", "parent": 16},
    18: {"name": "EYE_LEFT", "parent": 16},
    19: {"name": "EAR_LEFT", "parent": 16},
    20: {"name": "EYE_RIGHT", "parent": 16},
    21: {"name": "EAR_RIGHT", "parent": 16},
}

joint_relations = joint_relations22

def create_dataset(inputs, targets, sequence_size=1):
    dataset = []

    for i in range(len(inputs) - sequence_size + 1):
        input_sequence = inputs[i:i+sequence_size]
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
num_labels = 3 * len(joint_relations)

# テストデータの読み込み
test_inputs = []
test_targets = []

test_dir = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/ScannedTwoRadars_tilt4"
for i in range(num_flames+1):
    file_name = f"{i}.json"
    file_path = os.path.join(test_dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            if data["KeyPoint"][0][6 * 3 + 2] > data["KeyPoint"][0][5 * 3 + 2] or data["KeyPoint"][0][13 * 3 + 2] > data["KeyPoint"][0][12 * 3 + 2]:
                batch_list = []
                for x in range(min(data["NumPoints"], num_points)):
                    batch_list.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][2][x],
                                       data["PointCloud"][3][x], data["PointCloud"][4][x]])
                batch_list += [[0, 0, 0, 0, 0]] * (num_points - len(batch_list))
                test_inputs.append(batch_list)

                keypoint_ids = [0, 1, 2, 3, 5, 6, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                # keypoint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                test_targets.append([coord for idx in keypoint_ids for coord in data["KeyPoint"][0][idx * 3:(idx + 1) * 3]])

print(len(test_inputs))
print(len(test_inputs[0]))
print(len(test_inputs[0][0]))
test_inputs = np.array(test_inputs)
test_targets = np.array(test_targets)

min_values = test_targets.min(axis=0)
max_values = test_targets.max(axis=0)
#print(min_values.mean())
#print(max_values.mean())
test_targets = (test_targets - min_values) / (max_values - min_values)

test_inputs = torch.from_numpy(test_inputs)
test_targets = torch.from_numpy(test_targets)
test_inputs = test_inputs.to(torch.float32)
test_targets = test_targets.to(torch.float32)

data = TensorDataset(test_inputs, test_targets)
dataset = create_dataset(data.tensors[0], data.tensors[1])
num_iterations_per_epoch = len(dataset) // batch_size

# 学習済みモデルの読み込み
model = PointNetPoseCNN(num_points, num_labels).cuda()
model.load_state_dict(torch.load("C:/Users/kodam/PycharmProjects/mmModels/Models/SimpleModel/saved_model_20.pth"))


mae_values = []
mae_values_ind = []

fig = plt.figure(figsize=(18, 12))
fig.suptitle('KeyPoints Plot of Predicted and True Values')

# 3D グラフの設定
ax_output = fig.add_subplot(121, projection='3d')
ax_targets = fig.add_subplot(122, projection='3d')

Overall_mae = []
Azimuth_mae = []
Elevation_mae = []
Depth_mae = []
count = 0
for trial in range(5):
    for iteration in range(num_iterations_per_epoch):
        indices = random.sample(range(len(dataset)), 1)
        data = [dataset[i] for i in indices]

        #ts_inputs = torch.Tensor(ts_item[0]).unsqueeze(0).squeeze(dim=0).cuda()
        #ts_targets = torch.Tensor(ts_item[1]).unsqueeze(0).cuda()
        ts_inputs = torch.stack([torch.Tensor(item[0]) for item in data]).squeeze(dim=0).cuda()
        #print(ts_inputs)
        middle_index = len(data) // 2
        ts_targets = torch.Tensor(data[middle_index][1]).unsqueeze(0).cuda()

        model.eval()
        ts_output = model(ts_inputs)

        # 逆正規化された値でMAEを計算
        restored_output = (ts_output.squeeze().cpu().detach().numpy() * (max_values - min_values)) + min_values
        restored_targets_batch = (ts_targets.squeeze().cpu().detach().numpy() * (max_values - min_values)) + min_values
        mae = np.abs(restored_output - restored_targets_batch)
        #mae = ((restored_output - restored_targets_batch)**2)

        # 逆正規化なし
        #mae = np.abs(ts_output.squeeze().cpu().detach().numpy() - ts_targets.squeeze().cpu().detach().numpy())
        #mae = ((ts_output.squeeze().cpu().detach().numpy() - ts_targets.squeeze().cpu().detach().numpy())**2)

        if (mae[15]+mae[17])/2 < 120 and (mae[21]+mae[23])/2 < 120:
            count += 1

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
        ax_output.set_title('Prediction')
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
        plt.pause(0.001)

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

        #print('MAE on test data:', mae.mean())

        #print('Predictions:', ts_output)
        #print('True values:', ts_targets)

    # 各イテレーションのMAEの平均を計算
    average_pointwise_mae = np.mean(mae_values_ind, axis=0)

    Overall_mae.append(np.mean(mae_values))

    Azimuth_mae.append(average_pointwise_mae[::3])
    Elevation_mae.append(average_pointwise_mae[2::3])
    Depth_mae.append(average_pointwise_mae[1::3])

    print(f'Trial{trial+1} is Done!')

    # print('Average Pointwise MAE: ')
    # for coord_index in range(len(average_pointwise_mae)):
    #     coord_mae = average_pointwise_mae[coord_index]
    #     print(f'Coordinate {coord_index}: {coord_mae}')

Arm_Azimuth_ave = np.mean([Azimuth_mae[0][5], Azimuth_mae[0][7]])
Arm_Elevation_ave = np.mean([Elevation_mae[0][5], Elevation_mae[0][7]])
Arm_ave = np.mean([Arm_Azimuth_ave, Arm_Elevation_ave])

Leg_Azimuth_ave = np.mean(np.concatenate([Azimuth_mae[0][9:12], Azimuth_mae[0][13:16]]))
Leg_Elevation_ave = np.mean(np.concatenate([Elevation_mae[0][9:12], Elevation_mae[0][13:16]]))
Leg_ave = np.mean([Leg_Azimuth_ave, Leg_Elevation_ave])

Turn_ave = np.mean(np.concatenate([Azimuth_mae[0][4:16], Azimuth_mae[0][18:22]]))
print('==============================================')
print(count)
print(5*num_iterations_per_epoch)
print(f'Truth: {count/(5*(num_iterations_per_epoch+1))*100}%')
print('Overall MAE Average: ', np.mean(Overall_mae))
print('Azimuth MAE Average: ', np.mean(np.mean(Azimuth_mae, axis=1)))
print('Elevation MAE Average: ', np.mean(np.mean(Elevation_mae, axis=1)))
print('Depth MAE Average: ', np.mean(np.mean(Depth_mae, axis=1)))
print('Arm MAE Average: ', Arm_ave)
print('Leg MAE Average: ', Leg_ave)
print('Turn MAE Average: ', Turn_ave)
print('Azimuth MAE: ', Azimuth_mae[0])
print('Elevation MAE: ', Elevation_mae[0])
print('Depth MAE: ', Depth_mae[0])
x = Azimuth_mae[0]
y = Depth_mae[0]
z = Elevation_mae[0]
plt.figure(figsize=(10, 6))
plt.ylim(0, 450)
plt.plot(x, label='Azimuth')
plt.plot(y, label='Depth')
plt.plot(z, label='Elevation')
plt.xlabel('Joint Index')
plt.ylabel('Error(mm)')
plt.title('Mean Absolute Error')
plt.legend()
plt.show()

# Final plot
# plt.plot(mae_values, label='MAE')
# plt.xlabel('Frame')
# plt.ylabel('MAE Value')
# plt.title('Final MAE Plot')
# plt.legend()
# plt.show()
