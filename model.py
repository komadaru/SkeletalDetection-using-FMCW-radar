import torch
import torch.nn as nn
import torch.optim as optim

# NonLinear(Dense, ReLU, Batch Normalization)
class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.output_channels)
        )

    def forward(self, input_data):
        print(input_data.size())
        # if input_data.size(0) > 1:
        #
        #     input_data_batch = self.main(input_data).t()
        #     output_data_batch_normalized = nn.BatchNorm1d(input_data_batch.size(1))(input_data_batch)
        #     output = output_data_batch_normalized.t()
        #
        #     #output = nn.BatchNorm1d(self.output_channels)(self.main(input_data))
        # else:
        #     output = self.main(input_data)

        output = self.main(input_data)

        return output

# NonCNN(Dense, ReLU, Batch Normalization)
class NonCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(NonCNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Conv1d(self.input_channels, self.output_channels, 1),
            nn.ReLU(inplace=True),

        )

    def forward(self, input_data):
        print(self.input_channels)
        print(input_data.size())
        print(input_data)
        # print(self.input_channels)
        # print(input_data.size())
        # if input_data.size(0) > 1:
        #
        #     input_data_batch = self.main(input_data).t()
        #     output_data_batch_normalized = nn.BatchNorm1d(input_data_batch.size(1))(input_data_batch)
        #     output = output_data_batch_normalized.t()
        #
        #     # output = nn.BatchNorm1d(self.output_channels)(self.main(input_data))
        # else:
        #     output = self.main(input_data)

        output = self.main(input_data)

        return output

# NonLSTM(Dense, ReLU, BatchNormalization)
class NonLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(NonLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.LSTM(self.input_channels, self.hidden_channels),
            nn.Linear(self.hidden_channels, self.output_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.output_channels)
        )

    def forward(self, input_data):
        return self.main(input_data)

# MaxPooling層
class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        out = input_data.reshape(-1, self.num_channels, self.num_points)
        out = self.main(out)
        out = out.view(-1, self.num_channels)
        return out

class MaxPool1(nn.Module):
    def __init__(self):
        super(MaxPool2, self).__init__()

    def forward(self, input_data):
        out = torch.max(input_data, 1, keepdim=True)[0]
        #out = out.view(-1, 1024)
        print(out)
        return out
class MaxPool2(nn.Module):
    def __init__(self):
        super(MaxPool2, self).__init__()

    def forward(self, input_data):
        print("==============")
        print(input_data.size())
        out = torch.max(input_data, 2, keepdim=True)[0]
        print(out.size())
        #out = out.view(-1, 1024)
        print(out)
        return out

# InputTNet(Transform層)
class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        # self.main = nn.Sequential(
        #     NonCNN(3, 64),
        #     NonCNN(64, 128),
        #     NonCNN(128, 1024),
        #     MaxPool(1024, self.num_points),
        #     NonCNN(1024, 512),
        #     NonCNN(512, 256),
        #     nn.Linear(256, 9)
        # )

        self.main = nn.Sequential(
            NonCNN(4, 64),
            NonCNN(64, 128),
            NonCNN(128, 1024),
            MaxPool2(),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 9)
        )

    # shape of input_data is (batchsize × num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 3, 3)
        out = torch.matmul(input_data.view(-1, self.num_points, 3), matrix)
        out = out.view(-1, 3)
        return out

# featureTNet(中間Transform層)
class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonCNN(64, 64),
            NonCNN(64, 128),
            NonCNN(128, 1024),
            MaxPool2(),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out

class PointNet1(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet1, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.num_points),
            NonCNN(4, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(p=0.3),
            NonLinear(512, 256),
            nn.Dropout(p=0.3),
            NonLinear(256, self.num_labels)
        )

    def forward(self, input_data):
        print(input_data.size())
        return self.main(input_data.permute(1, 0).unsqueeze(0))

class PointNet2(nn.Module):
    def __init__(self, num_points, num_labels):
        super(PointNet2, self).__init__()
        self.num_points = num_points
        self.num_labels = num_labels

        self.main = nn.Sequential(
            InputTNet(self.num_points),
            NonCNN(3, 64),
            NonCNN(64, 64),
            FeatureTNet(self.num_points),
            NonCNN(64, 64),
            NonCNN(64, 128),
            NonCNN(128, 1024),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_labels)
        )

    def forward(self, input_data):
        return self.main(input_data)