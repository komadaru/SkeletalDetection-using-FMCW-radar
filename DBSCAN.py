import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# クラスタリング結果とノイズ除去後のデータのプロット関数
# クラスタリング結果とノイズ除去後のデータのプロット関数
def plot_clusters_and_cleaned_data(data, labels, cleaned_data):
    fig = plt.figure(figsize=(12, 6))

    # オリジナルの3D点群をプロット
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(data[0, :], data[1, :], data[2, :], c='gray', marker='o', label='Original Data')
    ax1.set_title('Original 3D Point Cloud')

    # クラスタリング結果をプロット
    ax2 = fig.add_subplot(122, projection='3d')
    for label in set(labels):
        if label == -1:
            # ノイズの場合は別色でプロット
            ax2.scatter(data[0, labels == label], data[1, labels == label], data[2, labels == label],
                        c='black', marker='x', label='Noise')
        else:
            # クラスタごとに異なる色でプロット
            ax2.scatter(data[0, labels == label], data[1, labels == label], data[2, labels == label],
                        marker='o', label=f'Cluster {label}')

    ax2.set_title('Clustering Result')

    # ノイズ除去後のデータをプロット
    fig = plt.figure(figsize=(6, 6))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.scatter(cleaned_data[0, :], cleaned_data[1, :], cleaned_data[2, :], c='green', marker='o', label='Cleaned Data')
    ax3.set_title('Cleaned 3D Point Cloud')

    plt.show()

flames_data = []
file_path = f"C:/Users/kodam/PycharmProjects/mmModels/Datasets/TwoRadars_tilt4/502.json"
with open(file_path, "r", encoding="utf-8") as json_file:
    data = json.load(json_file)
    flames_data.append(data)

# PointCloudデータの抽出
point_cloud = np.array(flames_data[0]["PointCloud"])

# DBSCANの設定
eps = 400.0  # イプシロン（近傍の距離の閾値）
min_samples = 5  # クラスタとして認識する最小の点の数
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# DBSCANの実行
labels = dbscan.fit_predict(point_cloud.T)  # dataを転置して渡す
print(labels)

# ノイズ以外の点のインデックスを取得
non_noise_indices = np.where(labels != -1)[0]

# ノイズ以外の点の座標を取得
cleaned_data = point_cloud[:, non_noise_indices]

# プロット
plot_clusters_and_cleaned_data(point_cloud, labels, cleaned_data)
