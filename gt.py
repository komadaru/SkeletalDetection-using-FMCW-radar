import torch
import torch.nn as nn

# ダミーの画像データ（バッチサイズ1、縦横24ピクセル、3チャンネル）
image_data = torch.randn(1, 3, 24, 24)

print(image_data)

# Conv2dの定義
conv2d_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

# 畳み込みを適用
output_data = conv2d_layer(image_data)

print(output_data.shape)
