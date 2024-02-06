# このプログラムについて
## 1. DataMakeフォルダ
DataMakeフォルダには、生のデータからデータセットを作成するためのコードが含まれています。

ミリ波レーダーで収集した点群データとKinectで収集した骨格データを統合する処理が書かれています。
## 2. メインフォルダ
メインフォルダには、前処理を行うためのコードと、モデルの定義を行うためのコード、モデルの学習/検証とテストを行うためのコードが含まれています。

**※学習/検証/テストにはNVIDIA GPUが必要(本研究ではRTX3070を使用)**
## 3. 使用ライブラリ/ツール
matplotlib 3.8.2, numpy 1.26.4, scikit-learn 1.4.0, torch 2.2.0

Anaconda->CUDA

NVIDIA Driver

# 使用方法
## 1. DataMakeフォルダ
ミリ波レーダー2台の場合->making_dataset.py

ミリ波レーダー1台の場合->making_dataset2.py
### ・設定する変数
dir[1,2], skedir, savedir, start_time, end_time, delta_seconds, start_index[1,2], end_index[1,2]
#### - dir[1,2]
点群データのパスを指定
#### - skedir
骨格データのパスを指定
#### - savedir
データセットの保存先を指定
#### - start_time
タイムスタンプの先頭を指定
#### - end_time
タイムスタンプの最後尾を指定
#### - start_index[1,2]
点群データのjsonの先頭IDを指定
#### - end_index[1,2]
点群データのjsonの最後尾IDを指定
## 2. メインフォルダ
## 2-1. 前処理
確認用->DBSCAN.py

実際の処理->DBSCAN2.py
### ・設定する変数(DBSCAN2.py)
eps, min_samples, input_folder, output_folder
#### - eps
DBSCANのイプシロンを設定
#### - min_samples
DBSCANの最小点数を指定
#### - input_folder
DBSCANを行いたいデータセットを指定
#### - output_folder
保存先を指定
## 2-2. モデルの定義
各々PyTorchの文法に従って定義
## 2-3. 学習/検証
ここでは主に使用したmain2d.pyについて説明
### ・設定する変数
keypoint_ids, num_points, epochs, dir, output_folder
#### - keypoint_ids
骨格座標のIDを設定
#### - num_points
点群データの最大点数を指定
#### - epochs
エポック数を指定
#### - output_folder
学習済みモデルの保存先を指定
## 2-4. テスト
ここでは主に使用したmodel_test2d.pyについて説明

手を振る動作のみについてテスト->model_test2d_swap.py
### ・設定する変数
joint_relations, num_flames, num_points, model.load_state_dict内のパス,
#### - joint_relations
予測するキーポイントの数に応じて32, 22, 17を指定
#### - num_flames
テストデータセットのフレーム数を指定
#### - num_points
点群データの最大点数を指定
#### - model.load_state_dict内のパス
テストしたい学習済みモデルのパスを指定
## 2-5. その他
keyviz.py -> 点群データとkinectのキーポイント座標同じグラフに描画
