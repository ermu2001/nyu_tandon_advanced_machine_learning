import os
import os.path as osp
import sys
import tqdm
import numpy as np


import torch
import torch.nn as nn

from datasets import SPAMBASE_PATH
from experts import Expert
import config
from experts import TheDataset

class ExpertsAdvice():
    def __init__(self, config) -> None:
        device, dtype = 'cuda', torch.float32
        self.config = config
        self.lr = self.config.experts_advice.lr
        self.num_experts = config.num_experts
        self.experts = []
        for i in range(1, config.num_experts+1):
            expert = Expert(i, **config.experts.expert).to(device, dtype)
            expert.eval()
            load_path = osp.join(config.output_dir, f'experts_{i}', 'pytorch_model.bin')
            with open(load_path, 'rb') as f:
                state_dict = torch.load(f)
            expert.load_state_dict(state_dict)
            expert = expert.eval()
            self.experts.append(expert)

        # init transition matrix as uniform
        self.task_type = config.task_type
        if config.experts_advice.type == 'static':
            self.transition_matrix = np.diag(np.ones(config.num_experts))
        elif config.experts_advice.type == 'fixshare':
            self.alpha = config.experts_advice.alpha
            self.transition_matrix = np.ones((self.num_experts, self.num_experts)) * self.alpha / (self.num_experts - 1)
            np.fill_diagonal(self.transition_matrix, 1-self.alpha)
        else:
            raise NotImplementedError()
        
        # initialize distribution
        # self.distribution = np.exp(np.random.randn(self.num_experts))

        self.distribution = np.ones(self.num_experts)
        self.distribution = self.distribution / self.distribution.sum()

    def predict(self, x):
        predictions = np.array([self.expert_predict(x, i) for i in range(self.num_experts)])
        experts_advice = self.distribution * predictions
        experts_advice = experts_advice.sum()
        return experts_advice

    def expert_predict(self, x, i):
        device, dtype = 'cuda', torch.float32
        expert = self.experts[i]
        with torch.no_grad():
            predictions = expert(torch.from_numpy(x)[None, ...].to(device, dtype)).cpu().numpy().item() 
        return predictions


    def train(self, x, y):
        predictions = np.array([self.expert_predict(x, i) for i in range(self.num_experts)])
        y = np.array(y)[..., None]
        if self.task_type == 'classification':
            # predictions for chances that sample is classified as 1,
            predictions = 1 / (1 + np.exp(-predictions)) # sigmoid
            predictions = predictions.clip(1e-5, 1 - 1e-5) # in case log goes to inf
            losses = - (y * np.log(predictions) + (1 - y) * np.log(1-predictions)) # nll loss
        elif self.task_type == 'regression':
            losses = (y - predictions) ** 2 # mse loss
        else:
            raise NotImplementedError()
        updated_distribution = np.matmul(self.distribution * np.exp(- self.lr * losses), self.transition_matrix)
        updated_distribution = updated_distribution / updated_distribution.sum()
        self.distribution = updated_distribution
        return losses

def main(config):
    experts_advice=ExpertsAdvice(config)
    dataset = TheDataset(config.data_path, is_train=True, num_features=config.experts.expert.in_features)
    for i, sample in tqdm.tqdm(enumerate(dataset)):
        x, y = sample
        experts_advice.train(x, y)
        if i % 100 == 0:
            print(f'distribution of experts advice after seeing {i} samples:', experts_advice.distribution)

    eval_dataset = TheDataset(config.data_path, is_train=False, num_features=config.experts.expert.in_features)
    total, cost = len(eval_dataset), 0
    for sample in tqdm.tqdm(eval_dataset):
        x, y = sample
        pred_y = experts_advice.predict(x)
        if config.task_type == 'classification': 
            pred_y = int(pred_y > 0.5)
            if y != pred_y:
                cost += 1
        elif config.task_type == 'regression':
            cost += (pred_y - y) ** 2
        else:
            raise NotImplementedError()
        
    print(f'after iterating same training data--experts advice error on same eval data: {cost/total:.4f}')


if __name__ == '__main__':
    cfg_name = sys.argv[1]
    cfg = getattr(config, cfg_name)
    main(cfg)