import os.path as osp
import tqdm
import numpy as np


import torch
import torch.nn as nn

from datasets import SPAMBASE_PATH
from experts import Expert
from config import cloud_config
from experts import SpambaseDataset

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
 
            self.experts.append(expert)
        print(self.experts)

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
        return int(experts_advice > 0.5)

    def expert_predict(self, x, i):
        device, dtype = 'cuda', torch.float32
        expert = self.experts[i]
        with torch.no_grad():
            if self.task_type == 'classification':
                predictions = expert(torch.from_numpy(x).to(device, dtype)).cpu().numpy().item() 
                predictions = 1 / (1 + np.exp(-predictions)) # sigmoid
            else:
                raise NotImplementedError()
        return predictions


    def train(self, x, y):
        predictions = np.array([self.expert_predict(x, i) for i in range(self.num_experts)])
        
        if self.task_type == 'classification':
            losses = - np.log(predictions) if y == 1 else - np.log(1-predictions) # nll loss
        elif self.task_type == 'regression':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        updated_distribution = np.matmul(self.distribution * np.exp(- self.lr * losses), self.transition_matrix)
        updated_distribution = updated_distribution / updated_distribution.sum()
        self.distribution = updated_distribution

def main():
    experts_advice=ExpertsAdvice(cloud_config)
    dataset = SpambaseDataset(SPAMBASE_PATH)
    for i, sample in tqdm.tqdm(enumerate(dataset)):
        x, y = sample
        experts_advice.train(x, y)
        if i % 100 == 0:
            print(f'distribution of experts advice after seeing {i} samples:', experts_advice.distribution)

    eval_dataset = SpambaseDataset(SPAMBASE_PATH, is_train=False)
    total, correct = len(eval_dataset), 0
    for sample in tqdm.tqdm(eval_dataset):
        x, y = sample
        pred_y = experts_advice.predict(x)
        if y == pred_y:
            correct += 1
    print(f'after iterating same training data--experts advice acc on same eval data: {correct/total:.4f}')


if __name__ == '__main__':
    main()