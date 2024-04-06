

import itertools
from tqdm import tqdm
import numpy as np
import scipy.io

import torch
import torch.nn as nn


def load_data(filepath):

    # Load .mat file
    mat_data = scipy.io.loadmat(filepath)
    train_samples = [
        {
            "x": x,
            "y": y
        } for x, y in zip(mat_data['TrainingX'], mat_data['TrainingY'])
    ]
    test_samples = [
        {
            "x": x,
            "y": y
        } for x, y in zip(mat_data['TestX'], mat_data['TestY'])
    ]
    return train_samples, test_samples

# train_samples, test_samples = load_data('data1.mat')
# print(len(train_samples))
# print(len(test_samples))

class Model():
    eps = 1e-5
    def __init__(self, out_dim, train_data, loss_lambda = 0.5):
        
        in_dim, _ = train_data.shape
        # self.weights = np.random.randn(in_dim, out_dim)
        self.weights = np.zeros((in_dim, out_dim))
        self.loss_lambda = loss_lambda
        self.train_data = train_data
        # self.rbf_co = np.mean([np.sqrt(np.sum(np.square(x1 - x2))) for x1, x2 in tqdm(itertools.product(self.train_data, self.train_data), total=len(self.train_data) ** 2)])
        self.rbf_co = 10.077141124806595 # precomputed
        # self.hessian_matrix = np.identity(in_dim)
        print(self.rbf_co)

    def compute_k_vec(self, x):
        assert x.ndim==2
        n, dim = x.shape
        N, _ = self.train_data.shape
        # k = np.square(self.train_data[ None, :, :] - x[ :, None, :]) # N N dim
        # k = np.sqrt(np.sum(k, axis=-1))
        k = np.zeros((n, N))
        for i in range(n):
            k[i] = np.sqrt(np.sum(np.square(x[i] - self.train_data), axis=-1))

        k = np.exp( - k / self.rbf_co / 2)
        return k

    def get_loss(self, features, labels):
        if labels.ndim == 1:
            labels = labels[:, None]
        returns = tuple()
        bsz, dim = features.shape
        features = self.compute_k_vec(features)
        returns = (features, ) + returns # 5
        features = np.matmul(features, self.weights)
        returns = (features, ) + returns # 4
        features = labels * features
        returns = (features, ) + returns # 3
        features =  1 / ( 1 + np.exp(-features) ) # doesn't really matters if it overflows.
        returns = (features, ) + returns # 2
        features = np.log(np.clip(features, self.eps, 1))
        returns = (features, ) + returns # 1
        loss = - np.sum(features, axis=0) # + self.loss_lambda * np.matmul(self.weights.T, self.weights)
        returns = (loss, ) + returns # 0
        return returns

    def predict(self, features,):
        bsz, dim = features.shape
        features = self.compute_k_vec(features)
        pred = np.matmul(features, self.weights)
        return pred
    
    def compute_gradient(self, features, labels, returns):
        bsz = features.shape[0]

        grad_loss2act = -1 / bsz # 0
        grad_loss2proj = grad_loss2act * (1 - returns[2])
        grad_loss2proj = labels * grad_loss2proj
        grad_loss2weights = returns[5].T @ grad_loss2proj
        
        grad_loss2weights = grad_loss2weights + self.loss_lambda * 2 * self.weights
        # grad_l2_norm = np.sqrt(np.sum(np.square(grad_loss2weights)))
        # grad_l1_norm = np.sum(np.abs(grad_loss2weights))
        # return grad_l1_norm
        return grad_loss2weights

    def sgd(self, features, labels, returns, lr):
        grad = self.compute_gradient(features, labels, returns)
        step = lr * grad
        self.weights = self.weights - step
        return np.sqrt(np.sum(np.square(grad)))

    def bfgs(self, features, labels, returns, lr):
        grad = self.sgd(features, labels, returns)

        if not hasattr(self, 'last_grad'):
            # first step
            self.hessian_matrix = np.identity(self.weights.shape[0])
            self.last_grad = grad
            self.last_weights = self.weights
            return
        

        # update the weights
        step = np.matmul( self.hessian_matrix, grad ) * lr * step
        self.last_weights = self.weights
        self.weights = self.weights - step
        
        # update hessian base on the last grad
        gamma = grad - self.last_grad
        delta = step
        
        denominator = np.dot(self.hessian_matrix @ gamma, gamma)
        beta = 1 + np.dot(gamma, delta.T) / denominator
        delta_hessian = self.hessian_matrix @ gamma @ delta.T / denominator \
                        - beta * self.hessian_matrix @ gamma @ gamma.T @ self.hessian_matrix / denominator

        self.hessian_matrix + delta_hessian

        return        
    

class ModelTorch(nn.Module):
    def __init__(self, out_dim, train_data, loss_lambda = 0.5):
        super().__init__()
        self.train_data = torch.Tensor(train_data)
        in_dim, _ = train_data.shape
        # self.weights = np.random.randn(in_dim, out_dim)
        self.weights = nn.Parameter(torch.zeros((in_dim, out_dim)))
        self.loss_lambda = loss_lambda
        self.train_data = train_data
        # self.rbf_co = np.mean([np.sqrt(np.sum(np.square(x1 - x2))) for x1, x2 in tqdm(itertools.product(self.train_data, self.train_data), total=len(self.train_data) ** 2)])
        self.rbf_co = 10.077141124806595 # precomputed
        # self.hessian_matrix = np.identity(in_dim)
        print(self.rbf_co)

    @torch.no_grad()
    def compute_k_vec(self, x):
        assert x.ndim==2
        n, dim = x.shape
        N, _ = self.train_data.shape
        # k = np.square(self.train_data[ None, :, :] - x[ :, None, :]) # N N dim
        # k = np.sqrt(np.sum(k, axis=-1))
        k = torch.zeros((n, N))
        for i in range(n):
            k[i] = torch.sqrt(torch.sum(torch.square(x[i] - self.train_data), axis=-1))

        k = np.exp( - k / self.rbf_co / 2)
        return k

    def forward(self, features):
        bsz, dim = features.shape
        features = self.compute_k_vec(features)
        pred = torch.matmul(features, self.weights)
        return pred