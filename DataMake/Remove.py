import os
import json

def remove_files_with_none(folder_path):
    files = sorted(os.listdir(folder_path))

    for filename in files:
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        now_time = data.get("Now_time")

        if now_time is None:
            print(file_path)
            os.remove(file_path)

        if now_time == "None":
            print(file_path)
            os.remove(file_path)

# フォルダのパスを指定
folder_path = "C:/Users/kodam/PycharmProjects/mmModels/DataMake/RawData/mmData1223_rm/OneRadar_fixed3"

# ファイルの削除と再連番化を実行
remove_files_with_none(folder_path)
