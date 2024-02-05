import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Num_channels = 5
Sequence_size = 1

# NonLinear(Dense, ReLU, Batch Normalization)
class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_data):
        return self.main(input_data)

class NonLinearBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinearBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            NonLinear(input_channels, output_channels),
            NonLinear(output_channels, output_channels),
            NonLinear(output_channels, output_channels),
        )

    def forward(self, input_data):
        return self.main(input_data)

class NonLinearForFlat(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinearForFlat, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_data):
        out = self.main(input_data)
        #return F.normalize(out.unsqueeze(0), p=2, dim=1).squeeze(0)
        return out

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=1, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        x = self.conv(x.unsqueeze(0).permute(0, 2, 1)).squeeze(0).permute(1, 0)
        if torch.isnan(x).any():
            print("Detected NaNX!!")
            print(x)
        x = self.relu(x)
        if torch.isnan(x).any():
            print("Detected NaNY!!")
        #x = self.batch_norm(x)
        if torch.isnan(x).any():
            print("Detected NaNZ!!")
        return x

class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNBlock, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(output_channels, output_channels, kernel_size=1, stride=1, padding=0),
            ConvBlock(output_channels, output_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.main(x)

# MaxPooling層
class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        out = input_data.contiguous().view(-1, self.num_channels, self.num_points)
        #out = input_data.permute(0, 2, 1).contiguous()
        out = self.main(out)
        out = out.view(-1, self.num_channels)
        return out

# InputTNet(Transform層)
class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(Num_channels, 64),
            NonLinear(64,128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, Num_channels**2)
        )

    # shape of input_data is (batchsize × num_points, channel)
    def forward(self, input_data):
        input_data = input_data.view(-1, Num_channels)
        matrix = self.main(input_data).view(Sequence_size, Num_channels, Num_channels)
        input_data = input_data.view(Sequence_size, self.num_points, Num_channels)
        out = torch.matmul(input_data, matrix)
        out = out.view(-1, Num_channels)
        return out

class InputTNetForCNN(nn.Module):
    def __init__(self, num_points):
        super(InputTNetForCNN, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            CNNBlock(Num_channels, 64),
            CNNBlock(64,128),
            CNNBlock(128, 1024),
            MaxPool(1024, self.num_points),
            CNNBlock(1024, 512),
            CNNBlock(512, 256),
            nn.Linear(256, Num_channels**2)
        )

        self.cnn1 = CNNBlock(Num_channels, 64)
        self.cnn2 = CNNBlock(64,128)
        self.cnn3 = CNNBlock(128, 1024)
        self.maxpool = MaxPool(1024, self.num_points)
        self.cnn4 = CNNBlock(1024, 512)
        self.cnn5 = CNNBlock(512, 256)
        self.fc = nn.Linear(256, Num_channels**2)

    # shape of input_data is (batchsize × num_points, channel)
    def forward(self, input_data):
        input_data = input_data.view(-1, Num_channels)

        x = self.cnn1(input_data)
        if torch.isnan(x).any():
            print("Detected NaNA!!")
            print(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.maxpool(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.fc(x)

        matrix = x.view(Sequence_size, Num_channels, Num_channels)
        input_data = input_data.view(Sequence_size, self.num_points, Num_channels)
        out = torch.matmul(input_data, matrix)
        out = out.view(-1, Num_channels)
        return out

# featureTNet(中間Transform層)
class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        point_feature = out
        return out

class FeatureTNetForCNN(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNetForCNN, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            CNNBlock(64, 64),
            CNNBlock(64, 128),
            CNNBlock(128, 1024),
            MaxPool(1024, self.num_points),
            CNNBlock(1024, 512),
            CNNBlock(512, 256),
            nn.Linear(256, 4096)
        )

    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        point_feature = out
        return out

class PointNetPose(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetPose, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = NonLinear(3, 64)
        self.fc2 = NonLinear(64, 64)
        self.fc3 = NonLinear(64, 64)
        self.fc4 = NonLinear(64, 128)
        self.fc5 = NonLinear(128, 1024)

        self.ITNet = InputTNet(self.num_points)
        self.FTNet = FeatureTNet(self.num_points)

        self.MPool = MaxPool(1024, 5 * self.num_points)

        self.fc6 = NonLinear(1088, 512)
        self.fc7 = NonLinear(512, 256)
        self.fc8 = NonLinear(256, 128)

        self.fc9 = NonLinearForFlat(5*650*128, 512)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data):
        x = self.ITNet(input_data)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.FTNet(x)
        point_feature = x
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.MPool(x)
        x = torch.tile(x, (5*self.num_points, 1))
        cat_mat = torch.cat((point_feature, x), dim=1)
        x = self.fc6(cat_mat)
        x = self.fc7(x)
        x = self.fc8(x)
        x = x.flatten()
        x = self.fc9(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc10(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc11(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc12(x)
        final_output = torch.unsqueeze(x, 0)

        return final_output

class PointNetPose2(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetPose2, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = NonLinear(4, 64)
        self.fc2 = NonLinear(64, 64)
        self.fc3 = NonLinear(64, 64)
        self.fc4 = NonLinear(64, 128)
        self.fc5 = NonLinear(128, 1024)

        self.ITNet = InputTNet(self.num_points)
        self.FTNet = FeatureTNet(self.num_points)

        self.MPool = MaxPool(1024, 5 * self.num_points)

        self.fc6 = NonLinear(1088, 512)
        self.fc7 = NonLinear(512, 256)
        self.fc8 = NonLinear(256, 128)

        self.fc9 = NonLinearForFlat(5*550*128, 512)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data):
        x = self.ITNet(input_data)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.FTNet(x)
        point_feature = x
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.MPool(x)
        x = torch.tile(x, (5*self.num_points, 1))
        cat_mat = torch.cat((point_feature, x), dim=1)
        x = self.fc6(cat_mat)
        x = self.fc7(x)
        x = self.fc8(x)
        x = x.flatten()
        x = self.fc9(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc10(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc11(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc12(x)
        final_output = torch.unsqueeze(x, 0)

        return final_output

class PointNetPoseCNN(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetPoseCNN, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = CNNBlock(Num_channels, 64)
        self.fc2 = CNNBlock(64, 64)
        self.fc3 = CNNBlock(64, 64)
        self.fc4 = CNNBlock(64, 128)
        self.fc5 = CNNBlock(128, 1024)

        self.ITNet = InputTNetForCNN(self.num_points)
        self.FTNet = FeatureTNetForCNN(self.num_points)

        self.MPool = MaxPool(1024, Sequence_size * self.num_points)

        self.fc6 = CNNBlock(1088, 512)
        self.fc7 = CNNBlock(512, 256)
        self.fc8 = CNNBlock(256, 128)

        self.fc9 = NonLinearForFlat(Sequence_size*550*128, 512)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data):
        x = self.ITNet(input_data)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc4(x)
        x = x.flatten()
        x = self.fc9(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc10(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc11(x)
        x = nn.Dropout(p=0.3)(x)
        final_output = self.fc12(x)

        return final_output

class PointNetmmPoseLinear(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetmmPoseLinear, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = NonLinearBlock(Num_channels, 64)
        self.fc2 = NonLinearBlock(64, 64)
        self.fc3 = NonLinearBlock(64, 64)
        self.fc4 = NonLinearBlock(64, 128)
        self.fc5 = NonLinearBlock(128, 1024)

        self.ITNet = InputTNet(self.num_points)
        self.FTNet = FeatureTNet(self.num_points)

        self.MPool = MaxPool(1024, Sequence_size * self.num_points)

        self.fc6 = NonLinearBlock(1088, 512)
        self.fc7 = NonLinearBlock(512, 256)
        self.fc8 = NonLinearBlock(256, 128)

        self.fc9 = NonLinearForFlat(Sequence_size * 550 * 128, 512)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data_xy, input_data_yz):
        xy = self.ITNet(input_data_xy)
        yz = self.ITNet(input_data_yz)
        # zx = self.ITNet(input_data_zx)
        xy = self.fc1(xy)
        yz = self.fc1(yz)
        # zx = self.fc1(zx)
        # xy = self.fc1(input_data_xy.view(-1, Num_channels))
        # yz = self.fc1(input_data_yz.view(-1, Num_channels))
        xy = self.fc2(xy)
        yz = self.fc2(yz)
        # zx = self.fc2(zx)
        ####################################
        # xy = self.fc3(xy)
        # yz = self.fc3(yz)
        ####################################
        cat_mat = torch.cat((xy, yz), dim=1)
        out = cat_mat.flatten(start_dim=0, end_dim=-1)
        out = self.fc9(out)
        out = self.fc10(out)
        out = self.fc11(out)
        final_output = self.fc12(out)

        return final_output

class PointNetmmPoseCNN(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetmmPoseCNN, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = CNNBlock(Num_channels, 64)
        self.fc2 = CNNBlock(64, 64)
        self.fc3 = CNNBlock(64, 64)
        self.fc4 = CNNBlock(64, 128)
        self.fc5 = CNNBlock(128, 1024)

        self.ITNet = InputTNetForCNN(self.num_points)
        self.FTNet = FeatureTNetForCNN(self.num_points)

        self.MPool = MaxPool(1024, Sequence_size * self.num_points)

        self.fc6 = CNNBlock(1088, 512)
        self.fc7 = CNNBlock(512, 256)
        self.fc8 = CNNBlock(256, 128)

        self.fc9 = NonLinearForFlat(Sequence_size*300*128, 512)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

        # self.fcA = NonLinearBlock(Num_channels, 32)
        # self.fcB = NonLinearBlock(32, 32)
        #
        # self.ITNet2 = InputTNet(self.num_points)

    def forward(self, input_data_xy, input_data_yz):
        xy = self.ITNet(input_data_xy)
        yz = self.ITNet(input_data_yz)
        xy = self.fc1(xy)
        yz = self.fc1(yz)
        xy = self.fc2(xy)
        yz = self.fc2(yz)

        # xy2 = self.ITNet2(input_data_xy)
        # yz2 = self.ITNet2(input_data_yz)
        # xy2 = self.fcA(xy2)
        # yz2 = self.fcA(yz2)
        # xy2 = self.fcB(xy2)
        # yz2 = self.fcB(yz2)

        cat_mat = torch.cat((xy, yz), dim=1)
        out = cat_mat.flatten(start_dim=0, end_dim=-1)
        out = self.fc9(out)
        out = self.fc10(out)
        out = self.fc11(out)
        final_output = self.fc12(out)

        return final_output

class PointNetmmPoseCNN2(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetmmPoseCNN2, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = CNNBlock(Num_channels, 64)
        self.fc2 = CNNBlock(64, 64)
        self.fc3 = CNNBlock(64, 64)
        self.fc4 = CNNBlock(64, 128)
        self.fc5 = CNNBlock(128, 1024)

        self.ITNet = InputTNetForCNN(self.num_points)
        self.FTNet = FeatureTNetForCNN(self.num_points)

        self.MPool = MaxPool(1024, Sequence_size * self.num_points)

        self.fc6 = CNNBlock(1088, 512)
        self.fc7 = CNNBlock(512, 256)
        self.fc8 = CNNBlock(256, 128)

        self.fc9 = NonLinearForFlat(Sequence_size*550*320, 512)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data_xyL, input_data_xyR, input_data_yz):
        xyL = self.ITNet(input_data_xyL)
        xyR = self.ITNet(input_data_xyR)
        yz = self.ITNet(input_data_yz)
        xyL = self.fc1(xyL)
        xyR = self.fc1(xyR)
        yz = self.fc1(yz)
        xyL = self.fc2(xyL)
        xyR = self.fc2(xyR)
        yz = self.fc2(yz)
        xyL = self.fc4(xyL)
        xyR = self.fc4(xyR)
        cat_mat = torch.cat((xyL, xyR, yz), dim=1)
        out = cat_mat.flatten(start_dim=0, end_dim=-1)
        out = self.fc9(out)
        out = self.fc10(out)
        out = self.fc11(out)
        final_output = self.fc12(out)

        return final_output

class PointNetPoseLSTM(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNetPoseLSTM, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.fc1 = CNNBlock(Num_channels, 64)
        self.fc2 = CNNBlock(64, 64)
        self.fc3 = CNNBlock(64, 64)
        self.fc4 = CNNBlock(64, 128)
        self.fc5 = CNNBlock(128, 1024)

        self.ITNet = InputTNetForCNN(self.num_points)
        self.FTNet = FeatureTNetForCNN(self.num_points)

        self.MPool = MaxPool(1024, Sequence_size * self.num_points)

        self.fc6 = CNNBlock(1088, 512)
        self.fc7 = CNNBlock(512, 256)
        self.fc8 = CNNBlock(256, 128)

        self.fc9 = nn.LSTM(Sequence_size*600*128, 512, batch_first=True)
        self.fc10 = NonLinearForFlat(512, 256)
        self.fc11 = NonLinearForFlat(256, 128)
        self.fc12 = NonLinearForFlat(128, self.num_labels)

    def forward(self, input_data):
        x = self.ITNet(input_data)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.FTNet(x)
        point_feature = x
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.MPool(x)
        x = torch.tile(x, (Sequence_size*self.num_points, 1))
        cat_mat = torch.cat((point_feature, x), dim=1)
        x = self.fc6(cat_mat)
        x = self.fc7(x)
        x = self.fc8(x)
        x = x.flatten()
        x = self.fc9(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc10(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc11(x)
        x = nn.Dropout(p=0.3)(x)
        x = self.fc12(x)
        final_output = torch.unsqueeze(x, 0)

        return final_output