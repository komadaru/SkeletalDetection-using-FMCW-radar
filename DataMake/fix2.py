import json
import numpy as np
import os

angle_lower = 0
height_lower = 1.4
depth_lower = -0.18

def rotate_elevation(points, angle):
    # elevation方向に回転する行列
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
        [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
    ])

    # 点群データに回転を適用
    rotated_points = np.dot(rotation_matrix, points[:3, :])

    # 各点のvおよびsnrの情報を残す
    rotated_points = np.vstack((rotated_points, points[3:, :]))

    return rotated_points

def add_height(points, height):
    # z座標に高さを足す
    points[2, :] += height

    return points

def add_depth(points, depth):
    points[1, :] += depth

    return points

def process_and_save_lower(file_path, angle, height, depth):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # 点群データの抽出
    point_cloud = np.array(data["PointCloud"])

    # 他のデータも抽出
    targets = np.array(data["Targets"])
    indexes = data["Indexes"]
    num_points = data["NumPoints"]
    num_targets = data["NumTargets"]
    frame_num = data["FrameNum"]
    fail = data["Fail"]
    classifier_output = data["ClassifierOutput"]
    now_time = data["Now_time"]

    # ① elevation方向に15度傾ける
    point_cloud = rotate_elevation(point_cloud, 0)
    point_cloud = add_height(point_cloud, 0)

    # ② レーダーの高さと深さを足す
    point_cloud = add_height(point_cloud, height)

    point_cloud = add_depth(point_cloud, depth)

    # ③ elevation方向に-15度傾ける
    point_cloud = rotate_elevation(point_cloud, -angle)

    # 結果を新しいJSONファイルに書き出す
    result_data = {
        "PointCloud": point_cloud.tolist(),
        "Targets": targets.tolist(),
        "Indexes": indexes,
        "NumPoints": num_points,
        "NumTargets": num_targets,
        "FrameNum": frame_num,
        "Fail": fail,
        "ClassifierOutput": classifier_output,
        "Now_time": now_time
    }
    # Save the updated data to a new JSON file
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(result_data, output_file)


input_folder = "C:/Users/kodam/PycharmProjects/mmModels/DataMake/RawData/mmData1223_rm/OneRadar3/"
output_folder = "C:/Users/kodam/PycharmProjects/mmModels/DataMake/RawData/mmData1223_rm/OneRadar_fixed3/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        print(file_path)
        process_and_save_lower(file_path, angle_lower, height_lower, depth_lower)