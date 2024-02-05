import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, input_data):
        x = self.conv(input_data)
        # x = self.conv2(x)
        # x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class NonLinearForFlat(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinearForFlat, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    def forward(self, input_data):
        return self.main(input_data)

class mmPose(nn.Module):
    def __init__(self, im_size, num_channels, num_labels):
        super(mmPose, self).__init__()
        self.im_size = im_size
        self.num_channels = num_channels
        self.num_labels = num_labels

        self.cnn1 = ConvBlock(self.num_channels, 16, 3)
        self.cnn2 = ConvBlock(16, 32, 3)
        self.cnn3 = ConvBlock(32, 64, 3)

        self.fc1 = NonLinearForFlat(24*24*128, 512)
        self.fc2 = NonLinearForFlat(512, 256)
        self.fc3 = NonLinearForFlat(256, 128)
        self.fc4 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data_xy, input_data_xz):
        y = self.cnn1(input_data_xy)
        y = self.cnn2(y)
        y = self.cnn3(y)
        z = self.cnn1(input_data_xz)
        z = self.cnn2(z)
        z = self.cnn3(z)
        cat_mat = torch.cat((y, z), dim=1)
        cat_mat = cat_mat.permute(0, 2, 3, 1)
        out = cat_mat.flatten() # 平坦化
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        final_output = self.fc4(out)

        return final_output

class simplePose(nn.Module):
    def __init__(self, im_size, num_channels, num_labels):
        super(simplePose, self).__init__()
        self.im_size = im_size
        self.num_channels = num_channels
        self.num_labels = num_labels

        self.cnn1 = ConvBlock(self.num_channels, 64, 3)
        self.cnn2 = ConvBlock(64, 64, 3)
        self.cnn3 = ConvBlock(64, 64, 3)

        self.fc1 = NonLinearForFlat(24*24*64, 512)
        self.fc2 = NonLinearForFlat(512, 256)
        self.fc3 = NonLinearForFlat(256, 128)
        self.fc4 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data):
        x = self.cnn1(input_data)
        x = self.cnn2(x)
        x = self.cnn3(x)
        out = x.flatten() # 平坦化
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        final_output = self.fc4(out)

        return final_output