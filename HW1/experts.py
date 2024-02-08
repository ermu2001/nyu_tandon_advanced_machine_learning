import itertools
import os
import os.path as osp
from datasets import (
    load_spambase,
    SPAMBASE_PATH
)
import tqdm
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch.utils.data import Dataset
import torch.utils.data



class SpambaseDataset():
    num_features=57
    def __init__(self, path, is_train=True):
        data = load_spambase(path)
        if is_train:
            self.samples = data['train']
        else:
            self.samples = data['val']

    def __getitem__(self, index):
        sample = self.samples[index]
        featrue = sample[:self.num_features].astype(np.float16)
        cls = sample[self.num_features].astype(np.int8)
        return featrue, cls

    def __len__(self):
        return self.samples.shape[0]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.00)

class Expert(nn.Module):
    out_features=1
    def __init__(self, num_layers, in_features=57, hidden_size=16):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        if num_layers > 1:
            self.hidden_size = hidden_size
            self.input_layer = nn.Sequential(nn.Linear(in_features, hidden_size), nn.ReLU())
            self.output_layer = nn.Linear(hidden_size, self.out_features)
            layers = itertools.chain.from_iterable(
                (nn.Linear(hidden_size, hidden_size), nn.ReLU()) for i in range(num_layers-2)
            )
            self.layers = nn.ModuleList(list(layers))
        else:
            self.hidden_size = 0
            self.input_layer = nn.Sequential(nn.Linear(in_features, self.out_features), nn.ReLU())
            self.output_layer = nn.Identity()
            self.layers = nn.ModuleList([])

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.layers[1:-1]:
            x = layer(x)
        x = self.output_layer(x)
        return x  


def train_one_expert(args, num_layers, save_dir):
    device, dtype = 'cuda', torch.float32
    os.makedirs(save_dir, exist_ok=True)
    dataset = SpambaseDataset(SPAMBASE_PATH)
    eval_dataset = SpambaseDataset(SPAMBASE_PATH, is_train=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16)
    expert = Expert(num_layers=num_layers, **args.expert).to(device=device, dtype=dtype)
    expert.apply(init_weights)
    optim = torch.optim.SGD(expert.parameters(), lr=args.lr )
    for epoch in range(args.epoch):
        tbar = tqdm.tqdm(dataloader)
        expert.train()
        for batch in dataloader:
            x, y = batch
            x = x.to(device, dtype)
            y = y.to(device, dtype=torch.long)
            optim.zero_grad()
            pred_hiddenstate = expert(x)
            pred_y = nn.functional.sigmoid(pred_hiddenstate)
            pred_y = torch.concatenate([1-pred_y, pred_y], dim=-1).clip(1e-5, 1 - 1e-5)
            pred_y = torch.log(pred_y)
            # onehot_y = nn.functional.one_hot(y)
            loss = nn.functional.nll_loss(pred_y, y)
            loss.backward()
            optim.step()
            tbar.set_description(f'{loss.detach().cpu().item()}')
            # print(f'{loss.detach().cpu().item()}')

        expert.eval()
        with torch.no_grad():
            num_total, num_correct = 0, 0
            for batch in eval_dataloader:
                x, y = batch
                x = x.to(device, dtype)
                y = y.to(device, dtype=torch.long)
                optim.zero_grad()
                pred_hiddenstate = expert(x)
                pred_y = nn.functional.sigmoid(pred_hiddenstate)
                pred_y = pred_y > 0.5
                pred_y = pred_y.squeeze(-1).long()
                correct = pred_y == y
                num_total += y.shape[0]
                num_correct += correct.sum().cpu().item()
        if num_correct / num_total > 1:
            print('what?')
        print(f'epoch {epoch}, expert with {num_layers} num_layers, eval acc {num_correct / num_total:.04}')

    param_path = osp.join(save_dir, 'pytorch_model.bin')
    with open(param_path, 'wb') as f:
        torch.save(expert.state_dict(), f)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for num_layers in range(1, args.num_experts+1):
        save_dir = osp.join(args.output_dir, f'experts_{num_layers}')
        train_one_expert(args=args.experts, num_layers=num_layers, save_dir=save_dir)





if __name__ == '__main__':
    from config import cloud_config
    print(cloud_config)
    main(cloud_config)

