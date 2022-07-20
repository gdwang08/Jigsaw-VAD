import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBranchNet(nn.Module):

    def __init__(self, time_length=7, num_classes=[127, 8]):
        super(WideBranchNet, self).__init__()
        
        self.time_length = time_length
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),

            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(self.time_length, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3)
            )
        self.max2d = nn.MaxPool2d(2, 2)
        self.classifier_1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[0])
        )
        self.classifier_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_classes[1])
        )
        
    
    def forward(self, x):
        out = self.model(x)
        out = out.squeeze(2)
        out = self.max2d(self.conv2d(out))
        out = out.view((out.size(0), -1))
        out1 = self.classifier_1(out)
        out2 = self.classifier_2(out)
        return out1, out2


if __name__ == '__main__':
    net = WideBranchNet(time_length=9, num_classes=[49, 81])
    x = torch.rand(2, 3, 7, 64, 64)
    out = net(x)
