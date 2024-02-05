import json
import csv
import os.path
from datetime import datetime, timedelta
import numpy as np

dir1 = "C:/Users/kodam/PycharmProjects/mmModels/DataMake/RawData/mmData1223_rm/OneRadar_fixed3/"
skedir = "C:/Users/kodam/PycharmProjects/mmModels/DataMake/RawData/Data1223/OneRadar3/20231223/"
savedir = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/OneRadar3/"

####################################### 1 ######################################
# フレームを決定し、リストへ格納
start_time = datetime(2023, 12, 23, 14, 34, 55) #変数
end_time = datetime(2023, 12, 23, 14, 41, 16) #変数
delta_seconds = 0.05 #フレーム間の秒数 #変数

time_list = []
current_time = start_time
while current_time <= end_time:
    time_list.append(current_time)
    current_time += timedelta(seconds=delta_seconds)
#print(time_list)

####################################### 2 ######################################
# jsonのNow_timeをリストに格納
start_index1 = 1 #変数
end_index1 = 7861 #変数

data1_times = []
for i in range(start_index1, end_index1 + 1):
    file_name = f"{i}.json"
    file_path = os.path.join(dir1, file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            # タイムゾーン情報を削除する場合^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # original_time = data["Now_time"]
            # removed_time = original_time[:-6]
            # data1_times.append([f"{i}.json", datetime.strptime(removed_time, '%Y-%m-%d %H:%M:%S.%f')])
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

            # タイムゾーン情報を削除しない場合^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            data1_times.append([f"{i}.json", datetime.strptime(data["Now_time"], '%Y-%m-%d %H:%M:%S.%f')])
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #print(f"Processing {file_name}")
    else:
        print(f"File {file_name} does not exist. Skipping.")

#骨格
skeleton_times = []
subfolders = [f.name for f in os.scandir(skedir) if f.is_dir()]
for subfolder in subfolders:
    hour = int(subfolder[:2])
    minute = int(subfolder[2:4])
    second = int(subfolder[4:6])
    microsecond = int(str(int(subfolder[6:])) + '000')
    datetime_obj = datetime(2023, 12, 23, hour, minute, second, microsecond)
    skeleton_times.append([subfolder, datetime_obj])

#print(data1_times)

####################################### 3 ######################################
# frameとjsonファイルの組み合わせをリストに格納[st_time, ed_time, [jsonfiles]]
frame_and_json_list1 = []
for time_idx in range(len(time_list)-1):
    st_time = time_list[time_idx]
    ed_time = time_list[time_idx + 1]
    frame_and_json_list1.append([st_time, ed_time])

    ts = []
    for dtime in data1_times:
        if st_time <= dtime[1] < ed_time:
            ts.append(dtime[0])
    if ts:
        frame_and_json_list1[time_idx].append(ts)
    else:
        frame_and_json_list1[time_idx].append('blank')

#骨格
frame_and_folder_list = []
for time_idx in range(len(time_list)-1):
    st_time = time_list[time_idx]
    ed_time = time_list[time_idx + 1]
    frame_and_folder_list.append([st_time, ed_time])

    ts = []
    for dtime in skeleton_times:
        if st_time <= dtime[1] < ed_time:
            ts.append(dtime[0])
    if ts:
        frame_and_folder_list[time_idx].append(ts)
    else:
        frame_and_folder_list[time_idx].append('blank')


####################################### 4 ######################################
# データセットとしてまとめる
json_number = 1

def rotate_around_z(points, angle_degrees):
    # Z軸周りの回転行列
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians), np.cos(angle_radians), 0],
        [0, 0, 1]
    ])

    # 点群データに回転を適用
    rotated_points = np.dot(rotation_matrix, points[:3, :])

    # 各点のvおよびsnrの情報を残す
    rotated_points = np.vstack((rotated_points, points[3:, :]))

    return rotated_points

for i in range(len(frame_and_json_list1)):
    PointCloud = [[], [], [], [], []]
    list1_is_blank = False
    if not frame_and_json_list1[i][2] == 'blank':
        for jsons in frame_and_json_list1[i][2]:
            file_path1 = dir1 + jsons
            with open(file_path1, 'r', encoding='utf8') as json_file:
                data = json.load(json_file)
                if data["NumPoints"] > 0:
                    for pidx in range(data["NumPoints"]):
                        PointCloud[0].append((data["PointCloud"][0][pidx])*1000)
                        PointCloud[1].append((data["PointCloud"][1][pidx])*1000)
                        PointCloud[2].append((data["PointCloud"][2][pidx])*1000)
                        PointCloud[3].append(data["PointCloud"][3][pidx])
                        PointCloud[4].append(data["PointCloud"][4][pidx])
    else:
        list1_is_blank = True

    #PointCloud = rotate_around_z(np.array(PointCloud), 180).tolist()

    KeyPoints = []
    kp_is_blank = False
    if not frame_and_folder_list[i][2] == 'blank':
        for folders in frame_and_folder_list[i][2]:
            ske_file_path = skedir + folders + "/0/pos.csv"
            with open(ske_file_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                ks = []
                for row in reader:
                    for idx in range(1, len(row) - 1):
                        if idx % 3 == 1:
                            ks.append(float(row[idx]))
                        elif idx % 3 == 2:
                            # rotated_y, rotated_z = rotate_yz_coordinates(float(row[idx])-1130, float(row[idx + 1]), 186)
                            # ks.append(rotated_y)
                            ks.append((float(row[idx]))-1200)
                        else:
                            # rotated_y, rotated_z = rotate_yz_coordinates(float(row[idx - 1])-1130, float(row[idx]), 186)
                            # ks.append(rotated_z)
                            ks.append(float(row[idx]))
                    KeyPoints.append(ks)
    else:
        kp_is_blank = True

    if not list1_is_blank and not kp_is_blank:

        NRKeyPoints = []
        for i in range(0, len(KeyPoints[0]), 3):
            x, y, z = KeyPoints[0][i:i+3]
            NRKeyPoints.extend([x, z, y])
        NRKeyPoints = [NRKeyPoints]

        # 座標系Bから最終的な座標系への変換行列
        transform_matrix_B_to_final = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(-6)), -np.sin(np.radians(-6))],
            [0, np.sin(np.radians(-6)), np.cos(np.radians(-6))]
        ])
        transform_matrix_B_to_final2 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

        middle_index = len(NRKeyPoints) // 2

        # KeyPointのX, Y, Z座標を取得
        kp_coordinates_B = np.array(NRKeyPoints[middle_index])  # 一つのサブリストにまとまっていると仮定
        kp_coordinates_B = kp_coordinates_B.reshape(-1, 3)  # キーポイントごとに3つの座標にまとめる

        # 座標変換を適用
        kp_coordinates_final = kp_coordinates_B
        #kp_coordinates_final = np.dot(kp_coordinates_B, transform_matrix_B_to_final.T)
        kp_coordinates_final = np.dot(kp_coordinates_final, transform_matrix_B_to_final2.T).tolist()

        # 変換後の座標を更新
        RKeyPoints = [NRKeyPoints[middle_index]]
        for idx, kp in enumerate(kp_coordinates_final):
            RKeyPoints[0][idx * 3] = kp[0]
            RKeyPoints[0][idx * 3 + 1] = kp[2]
            RKeyPoints[0][idx * 3 + 2] = kp[1]

        small_data = {
            "PointCloud": PointCloud,
            "NumPoints": len(PointCloud[0]),
            "KeyPoint": RKeyPoints
        }

        file_name = f"{json_number}.json"
        save_file_path = os.path.join(savedir, file_name)
        with open(save_file_path, "w") as json_file:
            json.dump(small_data, json_file)
        json_number += 1