import itertools
import os
import sys
import os.path as osp
from datasets import (
    load_preprocessed,
    SPAMBASE_PATH
)
import tqdm
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn

from torch.utils.data import Dataset
import torch.utils.data
import config


class TheDataset():
    def __init__(self, path, is_train=True, num_features=None):
        self.num_features = num_features
        data = load_preprocessed(path)
        if is_train:
            self.samples = data['train']
        else:
            self.samples = data['val']

    def __getitem__(self, index):
        sample = self.samples[index]
        featrue = sample[:self.num_features].astype(np.float32)
        cls = sample[self.num_features].astype(np.float32)
        return featrue, cls

    def __len__(self):
        return self.samples.shape[0]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.00)

class Expert(nn.Module):
    out_features=1
    def __init__(self, num_layers, in_features=None, hidden_size=16):
        super().__init__()
        self.num_layers = num_layers
        self.in_features = in_features
        if num_layers > 1:
            self.hidden_size = hidden_size
            self.input_layer = nn.Sequential(nn.Linear(in_features, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU())
            self.output_layer = nn.Linear(hidden_size, self.out_features)
            layers = itertools.chain.from_iterable(
                (nn.Linear(hidden_size, hidden_size), nn.ReLU()) for i in range(num_layers-2)
            )
            self.layers = nn.ModuleList(list(layers))
        else:
            self.hidden_size = 0
            self.input_layer = nn.Sequential(nn.Linear(in_features, self.out_features))
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
    dataset = TheDataset(args.data_path, is_train=True, num_features=args.experts.expert.in_features)
    eval_dataset = TheDataset(args.data_path, is_train=False, num_features=args.experts.expert.in_features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=16)
    expert = Expert(num_layers=num_layers, **args.experts.expert).to(device=device, dtype=dtype)
    expert.apply(init_weights)
    optim = torch.optim.Adam(expert.parameters(), lr=args.experts.lr)
    for epoch in range(args.experts.epoch):
        tbar = tqdm.tqdm(dataloader, disable=True)
        expert.train()
        for batch in dataloader:
            x, y = batch
            if x.shape[0] != 16:
                continue
            x = x.to(device, dtype)
            y = y.to(device, dtype)
            pred_hiddenstate = expert(x)
            if args.task_type == 'classification':
                y = y.to(dtype=torch.long)
                pred_y = nn.functional.sigmoid(pred_hiddenstate)
                pred_y = torch.concatenate([1-pred_y, pred_y], dim=-1).clip(1e-5, 1 - 1e-5)
                pred_y = torch.log(pred_y)
                # onehot_y = nn.functional.one_hot(y)
                loss = nn.functional.nll_loss(pred_y, y)
            elif args.task_type == 'regression':
                y = y[:, None]
                loss = nn.functional.mse_loss(pred_hiddenstate, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            tbar.set_description(f'loss: {loss.detach().cpu().item()}')
        expert.eval()
        with torch.no_grad():
            cost = 0
            num_total, num_correct = 0, 0
            for batch in eval_dataloader:
                x, y = batch
                x = x.to(device, dtype)
                y = y.to(device, dtype)
                pred_hiddenstate = expert(x)
                if args.task_type == 'classification':
                    y = y.to(dtype=torch.long)
                    pred_y = nn.functional.sigmoid(pred_hiddenstate)
                    pred_y = pred_y > 0.5
                    pred_y = pred_y.squeeze(-1).long()
                    correct = pred_y == y
                    num_total += y.shape[0]
                    num_correct += correct.sum().cpu().item()
                    cost = (num_total - num_correct) / num_total
                    assert 0<= cost <= 1
                elif args.task_type == 'regression':
                    y = y[:, None]
                    loss = nn.functional.mse_loss(pred_hiddenstate, y, reduce=True)
                    cost += loss.mean().cpu().item() / len(eval_dataloader)
        print(f'epoch {epoch}, expert with {num_layers} num_layers, eval error {cost:.04}')

    param_path = osp.join(save_dir, 'pytorch_model.bin')
    with open(param_path, 'wb') as f:
        torch.save(expert.state_dict(), f)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for num_layers in range(1, args.num_experts+1):
        save_dir = osp.join(args.output_dir, f'experts_{num_layers}')
        train_one_expert(args=args, num_layers=num_layers, save_dir=save_dir)





if __name__ == '__main__':
    cfg_name = sys.argv[1]
    cfg = getattr(config, cfg_name)
    main(cfg)

