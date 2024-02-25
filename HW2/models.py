import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_hidden_layers=1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers))
        self.linear3 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.act(x)
        x = self.linear3(x)
        return x
    
class LeNet(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.linear1 = nn.Linear(400, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        self.act = nn.Sigmoid()
        self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.act(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        x = self.linear3(x)
        return x
    
class ResNetCifarBlock(nn.Module):
    @classmethod
    def make_resblock_group(cls, input_nc, output_nc, n):
        blocks = []
        blocks.append(cls(input_nc, output_nc))
        for _ in range(1, n):
            blocks.append(cls(output_nc, output_nc))
        return nn.Sequential(*blocks)
    
    def __init__(self, input_nc, output_nc):
        super().__init__()
        stride = 1
        self.expand = False
        if input_nc != output_nc:
            assert input_nc * 2 == output_nc, 'output_nc must be input_nc * 2'
            stride = 2
            self.expand = True

        self.conv1 = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(output_nc)
        self.conv2 = nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(output_nc)

    def forward(self, x):
        xx = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(xx))
        if self.expand:
            x = F.interpolate(x, scale_factor=0.5, mode='nearest')  # subsampling
            zero = torch.zeros_like(x)
            x = torch.cat([x, zero], dim=1)  # option A in the original paper
        h = F.relu(y + x, inplace=True)
        return h

class ResNetCifar(nn.Module):

    def __init__(self, n):
        super().__init__()

        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.block1 = ResNetCifarBlock.make_resblock_group(16, 16, n)
        self.block2 = ResNetCifarBlock.make_resblock_group(16, 32, n)
        self.block3 = ResNetCifarBlock.make_resblock_group(32, 64, n)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  # global average pooling
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
