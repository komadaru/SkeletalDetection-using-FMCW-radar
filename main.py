from torch.utils.data import TensorDataset, random_split

from model2 import *

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import math

keypoint_ids = [0, 1, 2, 3, 5, 6, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
batch_size = 11023 #フレームの数
num_points = 550 #点群数
num_labels = 3 * len(keypoint_ids) #出力ラベル数
epochs = 100
windowsize = 1

inputs = []
targets = []

dir = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/ScannedCombinedTwoRadars_tilt"

skipped_count = 0
for i in range(batch_size):
    file_name = f"{i}.json"
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            # Check if all values in PointCloud are NaN
            if np.isnan(data["PointCloud"]).all():
                skipped_count+=1
                # Skip this frame
                continue

            batch_list = []
            snr_values = data["PointCloud"][4]
            min_snr = np.min(snr_values)
            max_snr = np.max(snr_values)
            normalized_snr = (snr_values - min_snr) / (max_snr - min_snr)
            data["PointCloud"][4] = normalized_snr
            for x in range(data["NumPoints"]):
                batch_list.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][2][x],
                                   data["PointCloud"][3][x], data["PointCloud"][4][x]])

            # Check if NumPoints is less than num_points
            if len(batch_list) < num_points:
                # Fill the remaining points with NaN
                batch_list += [[0, 0, 0, 0, 0]] * (num_points - len(batch_list))

            inputs.append(batch_list)

            targets.append([coord for idx in keypoint_ids for coord in data["KeyPoint"][0][idx*3:(idx+1)*3]])

'''
for i in range(batch_size):
    file_name = f"{i}.json"
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            # Check if all values in PointCloud are NaN
            if np.isnan(data["PointCloud"]).all():
                # Use a binary flag to indicate whether points exist or not
                has_points = 1
                point_coordinates = [[100, 100, 100, 100]] * num_points
            else:
                has_points = -1
                batch_list = []
                for x in range(data["NumPoints"]):
                    batch_list.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][2][x], data["PointCloud"][3][x]])

                # Fill the remaining points with NaN
                batch_list += [[100, 100, 100, 100]] * (num_points - len(batch_list))
                point_coordinates = batch_list

            # Add binary flag to the input
            inputs.append([[has_points] + coord for coord in point_coordinates])

            targets.append(data["KeyPoint"][0])
'''

'''
for i in range(batch_size):
    file_name = f"{i}.json"
    file_path = os.path.join(dir, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

            batch_list = []
            for x in range(data["NumPoints"]):
                batch_list.append([data["PointCloud"][0][x], data["PointCloud"][1][x], data["PointCloud"][2][x], data["PointCloud"][3][x]])
            batch_list += [[0, 0, 0, 0]] * (num_points - len(batch_list))
            inputs.append(batch_list)

            targets.append(data["KeyPoint"][0])
'''

print(len(inputs))
print(len(inputs[0]))
print(len(inputs[0][0]))
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

def create_dataset(inputs, targets, sequence_size=1, random_sampling=True):
    dataset = []

    if random_sampling:
        # ランダムなサンプリング
        for i in range(len(inputs) - sequence_size + 1):
            start_idx = random.randint(0, len(inputs) - sequence_size)
            input_sequence = inputs[start_idx:start_idx + sequence_size]
            target_sequence = targets[start_idx + sequence_size - 1]
            dataset.append((input_sequence, target_sequence))
    else:
        # 連続したサンプリング
        for i in range(len(inputs) - sequence_size + 1):
            input_sequence = inputs[i:i + sequence_size]
            target_sequence = targets[i + sequence_size - 1]
            dataset.append((input_sequence, target_sequence))

    return dataset

def main(input_data, target, num_epochs=epochs, batch_size=1, sequence_size=1):
    pointnet = PointNetPoseCNN(num_points, num_labels).cuda()

    #new_param = pointnet.state_dict()
    #new_param['main.0.main.6.bias'] = torch.eye(3, 3).view(-1)
    #new_param['main.3.main.6.bias'] = torch.eye(64, 64).view(-1)
    #pointnet.load_state_dict(new_param)

    data = TensorDataset(input_data, target)
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size

    #train_dataset, val_dataset = random_split(data, [train_size, val_size])
    train_dataset = TensorDataset(*data[:train_size])
    val_dataset = TensorDataset(*data[train_size:])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(pointnet.parameters(), lr=0.0001)

    plt.ion()
    fig, ax = plt.subplots()
    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, num_epochs+1):
        nan_det = False
        dataset = create_dataset(train_dataset.tensors[0], train_dataset.tensors[1], sequence_size)
        num_iterations_per_epoch = len(dataset) // batch_size
        epoch_loss = 0.0

        for iteration in range(num_iterations_per_epoch):
            batch_indices = random.sample(range(len(dataset)), 1)
            batch_data = [dataset[i] for i in batch_indices]

            batch_inputs = torch.stack([torch.Tensor(item[0]) for item in batch_data]).squeeze(dim=0).cuda()
            #print(batch_inputs)

            middle_index = len(batch_data) // 2
            batch_targets = torch.Tensor(batch_data[middle_index][1]).cuda()

            pointnet.zero_grad()
            output = pointnet(batch_inputs)

            error = criterion(output, batch_targets)
            if torch.isnan(error).any():
                nan_det = True
                break

            error.backward()

            optimizer.step()

            epoch_loss += error.item()

            if iteration % 100 == 0:
                print('Epoch: {} Iteration: {} Loss: {}'.format(epoch, iteration, error.item()))

        if nan_det == True:
            break
        epoch_loss /= num_iterations_per_epoch
        train_loss_list.append(epoch_loss)
        print('Epoch: {} Average Loss: {}'.format(epoch, epoch_loss))

        # Validation
        val_d = create_dataset(val_dataset.tensors[0], val_dataset.tensors[1], sequence_size)
        val_loss = 0.0

        with torch.no_grad():
            for val_item in val_d:
                val_inputs = torch.Tensor(val_item[0]).unsqueeze(0).cuda()
                val_targets = torch.Tensor(val_item[1]).cuda()

                val_output = pointnet(val_inputs)
                val_loss += criterion(val_output, val_targets).item()

        val_loss /= len(val_dataset)
        val_loss_list.append(val_loss)
        print('Epoch: {} Validation Average Loss: {}'.format(epoch, val_loss))

        # モデルの保存
        model_path = f"C:/Users/kodam/PycharmProjects/mmModels/Models/SimpleModel/saved_model_{epoch}.pth"
        torch.save(pointnet.state_dict(), model_path)

        ax.clear()
        ax.plot(train_loss_list, label='Train Loss')
        ax.plot(val_loss_list, label='Validation Loss')
        ax.legend()
        plt.pause(0.2)

    plt.ioff()
    plt.show()

    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSELoss')
    plt.legend()
    plt.show()

    '''
    for iteration, batch in enumerate(input_data):
        torch.autograd.set_detect_anomaly(True)

        pointnet.zero_grad()
        #print(batch.view(-1, 3))
        output = pointnet(batch)  # 予測された骨格キーポイント座標

        #print(target[iteration])
        error = criterion(output, torch.unsqueeze(target[iteration], 0))  # 正解の座標と予測の座標とのMAEが小さくなるようにする損失関数
        #print(error)
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
    '''

main(inputs, targets)