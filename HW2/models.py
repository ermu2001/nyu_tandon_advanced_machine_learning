import torch
import torch.nn as nn


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
        
class Resnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward(self):
        ...
