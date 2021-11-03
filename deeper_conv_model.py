# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         deeper_conv_model
# Description:
# Author:       梁超
# Date:         2021/11/3
# -------------------------------------------------------------------------------
import torch
import torch.nn as nn


class DCNet(nn.Module):
    def __init__(self, batch, in_channels, out_channels):
        super(DCNet, self).__init__()
        self.batch = batch
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(self.in_channels, 2 * self.in_channels)
        self.conv1 = nn.Conv1d(1, 16, (3,))
        self.mp1 = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(16, self.in_channels-1)

        self.fc2 = nn.Linear(self.in_channels-1, self.in_channels-1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=(3,),
                               padding=(1,), stride=(1,))
        self.mp2 = nn.MaxPool1d(2)
        self.bn2 = nn.BatchNorm1d(32, 16)

        self.fc3 = nn.Linear(16, 64)
        self.conv3 = nn.Conv1d(32, 64, (3,),
                               padding=(1,), stride=(1,))
        self.mp3 = nn.MaxPool1d(2)
        self.bn3 = nn.BatchNorm1d(64, 32)

        self.fc4 = nn.Linear(32, 16)
        self.conv4 = nn.Conv1d(64, 32, (3,),
                               padding=(1,), stride=(1,))
        self.mp4 = nn.MaxPool1d(2)
        self.bn4 = nn.BatchNorm1d(32, 8)

        self.pt = nn.Flatten()
        self.fc5 = nn.Linear(32*8, 2)
        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, self.in_channels))
        x = self.fc1(x)
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.conv3(x)
        x = self.mp3(x)
        x = self.bn3(x)

        x = self.fc4(x)
        x = self.conv4(x)
        x = self.mp4(x)
        x = self.bn4(x)
        x = self.pt(x)
        x = self.fc5(x)
        x = self.act(x)
        return x


if __name__ == '__main__':
    batch, in_features, out_features = 30, 33, 2
    model = DCNet(batch, in_features, out_features)
    x = torch.randn(batch, in_features)
    print(model(x).shape)
