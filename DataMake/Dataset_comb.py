import os
import shutil

def combine_folders(input_folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    current_frame_num = 1

    for folder_path in sorted(input_folders):
        # フォルダ内の.jsonファイルをソートして取得
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])

        for file_name in json_files:
            input_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(output_folder, f"{current_frame_num}.json")

            # ファイルをコピー
            shutil.copy(input_path, output_path)

            current_frame_num += 1
        print(current_frame_num)

    print("Combined folders into:", output_folder)


# フォルダを結合して新しいデータセットを作成
data1 = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/OneRadar"
data2 = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/OneRadar2"
#data3 = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/TwoRadars_tilt3"
input_folders = [data1, data2]
output_folder = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/CombinedOneRadar"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
combine_folders(input_folders, output_folder)
