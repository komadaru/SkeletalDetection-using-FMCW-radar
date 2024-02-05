import torch
import random

x = torch.randn(1, 10, 550)
print(x.size())
window_size = 10
windows = [x[:, :, i:i+window_size] for i in range(0, x.size(2)-window_size+1)]
print(len(windows))
print(len(windows[0]))
print(len(windows[0][0]))
print(len(windows[0][0][0]))

# batch_size = 64
# half_batch_size = 32
# num_points = 16
#
# normal_sampled = torch.randn(half_batch_size, num_points, 3)
# uniform_sampled = torch.rand(half_batch_size, num_points, 3)
#
# input_data = torch.cat((normal_sampled, uniform_sampled), dim=0)
#
# data_shuffle = torch.randperm(batch_size)
#
# print(len(input_data[data_shuffle]))
# print("================================")
# print(input_data[data_shuffle].view(-1, 3))


# import torch.nn as nn
#
# # 入力データをPyTorchのテンソルに変換
# input_data = torch.tensor([[1, 2, 3], [2, 3, 4], [5, 6, 7]], dtype=torch.float32)
#
# # nn.Linearの定義（入力次元: 3、出力次元: 64）
# linear_layer = nn.Linear(3, 64)
#
# # 入力データをFlattenしてnn.Linearに渡す
# output_data = linear_layer(input_data)
#
# print(output_data)
#
# output_data = nn.ReLU(inplace=True)(output_data)
#
# print(output_data)
#
# output_data = nn.BatchNorm1d(64)(output_data)
#
# print(output_data)
#
# out = output_data.view(-1, 64, 3)
# print(out)
#
# out = nn.MaxPool1d(3)(out)
# print(out)
#
# out = out.view(-1, 64)
#
# print(out)