
import pandas as pd
import pickle as pkl
import os
import os.path as osp
import numpy as np
import scipy
from scipy.io.arff import loadarff
from sklearn import datasets

from ucimlrepo import fetch_ucirepo 

CLOUD_PATH='data/cloud'
SPAMBASE_PATH='data/spambase'
EYE_PATH='data/EEG Eye State.arff'
def split_train_val(data, train_ratio=0.8, shuffle=True):
    num_samples = data.shape[0]
    num_train_samples = round(num_samples * train_ratio)
    if shuffle:
        np.random.shuffle(data)
    train_data, val_data = np.split(data, (num_train_samples, ), axis=0)
    return train_data, val_data

def load_spambase_from_uci():
    # fetch dataset 
    spambase = fetch_ucirepo(id=94) 
    
    # data (as pandas dataframes) 
    X = spambase.data.features 
    y = spambase.data.targets 
    
    # metadata 
    # print(spambase.metadata) 
    # variable information 
    # print(spambase.variables) 
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    data = np.concatenate([X_np, y_np], axis=1)
    return data

def load_spambase(path=SPAMBASE_PATH):
    with open(osp.join(SPAMBASE_PATH, 'train_val_dict.pkl'), 'rb') as f:
        train_val_dict = pkl.load(f)
    return train_val_dict

def load_eye(path=EYE_PATH):
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    np_data = df_data.to_numpy()
    data_dict = {
        'train': np_data
    }
    return data_dict

def load_iris():
    # Load a sample dataset (for example, the iris dataset)
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    y = y[:, None]
    data_dict = {
        'train': np.concatenate([X, y], axis=-1)
    }
    return data_dict


if __name__ == "__main__":
    # data = load_spambase_from_uci()
    # print(data.shape)
    # train_data, val_data = split_train_val(data)
    # print(train_data.shape)
    # print(val_data.shape)

    # os.makedirs(SPAMBASE_PATH, exist_ok=True)
    # train_val_dict = {
    #     'train': train_data,
    #     'val': val_data,
    # }
    # with open(osp.join(SPAMBASE_PATH, 'train_val_dict.pkl'), 'wb') as f:
    #     pkl.dump(train_val_dict, f)

    eye_dataset = load_eye()
    spambase_dataset = load_spambase()
    print(eye_dataset)
    print(spambase_dataset)
