import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np

start = 3000
keypoint_ids = [0, 1, 2, 3, 5, 6, 12, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26]

def rotate_yz_coordinates(y, z, angle_degree):
    angle_rad = np.radians(angle_degree)
    rotated_y = y * np.cos(angle_rad) - z * np.sin(angle_rad)
    rotated_z = y * np.sin(angle_rad) + z * np.cos(angle_rad)
    return rotated_y, rotated_z

# 20フレーム分のデータをロードする
frames_data = []
for frame_index in range(start, start+12):
    file_path = f"C:/Users/kodam/PycharmProjects/mmModels/Datasets/ScannedOneRadar3/{frame_index}.json"
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        frames_data.append(data)

# サブプロットのレイアウト調整
num_frames = len(frames_data)
num_cols = 4  # 1行に表示するサブプロットの数
num_rows = (num_frames + num_cols - 1) // num_cols  # 必要な行数

# グラフの作成
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), subplot_kw={'projection': '3d'})

# 各フレームごとにサブプロットにプロット
for frame_index, ax in enumerate(axes.flatten()):
    if frame_index < num_frames:
        data = frames_data[frame_index]

        # PointCloudをプロット
        scatter = ax.scatter(data["PointCloud"][0], data["PointCloud"][1], data["PointCloud"][2], c=data["PointCloud"][4], marker='o', cmap='viridis')
        fig.colorbar(scatter, ax=ax, label='Intensity')  # カラーバーを追加

        # 各座標を抽出してプロット
        keypoints = [coord for idx in keypoint_ids for coord in data["KeyPoint"][0][idx*3:(idx+1)*3]]
        for i in range(len(keypoints) // 3):
            x = keypoints[3 * i]
            y = keypoints[3 * i + 1]
            z = keypoints[3 * i + 2]

            #rotated_y, rotated_z = rotate_yz_coordinates(y, z, -96)
            #ax.scatter(x, rotated_y, rotated_z, label=f'Point {i + 1}', color='red', marker='x')

            ax.scatter(x, y, z, label=f'Point {i + 1}', color='red', marker='x')

            # 点の近くにラベルを表示
            #ax.text(x, rotated_y, rotated_z, f'{i}', color='red')
            ax.text(x, y, z, f'{i}', color='red')

        # グラフにラベルを追加
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 視点を変更してyz方向から見る
        #ax.view_init(azim=90, elev=0)

        # グラフタイトルにフレーム番号を表示
        ax.set_title(f'Frame {start + frame_index}')
        #ax.set_title(f'{data["Now_time"]}')

# レイアウト調整
plt.tight_layout()

# グラフを表示
plt.show()
