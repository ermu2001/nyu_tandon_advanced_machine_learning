import os.path as osp
import numpy as np
import pickle as pkl
from datas import DATA_FOLDER, GAUSSIAN_PATH, REGRESSION_PATH
def random_shuffle_split_save(samples, labels, train_ratio, path):
    permute = np.arange(samples.shape[0])
    np.random.shuffle(permute)
    samples = samples[permute]
    labels = labels[permute]
    print(samples)
    print(labels)
    num_train_samples = round(train_ratio * samples.shape[0])
    train_samples, train_labels = samples[:num_train_samples], labels[:num_train_samples]
    test_samples, test_labels = samples[num_train_samples:], labels[num_train_samples:]
    data = {
        'train_samples': train_samples,
        'train_labels': train_labels,
        'test_samples': test_samples,
        'test_labels': test_labels,
    }
    with open(path, 'wb') as f:
        pkl.dump(data, f)

def generate_gaussian():
    num_dim, num_classes, num_samples_per_class = 2, 4, 1250
    classes = np.arange(num_classes)
    means, covs = 5 * np.random.random((num_classes, num_dim)), 5 * np.random.random((num_classes, num_dim))
    gaussians = np.random.randn(num_classes, num_samples_per_class, num_dim)
    gaussians = (gaussians + means[:, None, :]) * covs[:, None, :]
    gaussians = gaussians.reshape((num_classes * num_samples_per_class, num_dim))
    labels = np.arange(num_classes).repeat(num_samples_per_class)
    random_shuffle_split_save(gaussians, labels, 0.9, path=GAUSSIAN_PATH)

def generate_regression():
    num_dim, num_samples = 2, 5000
    samples = np.random.random((num_samples, num_dim)) * 20 - 10 
    labels = samples[:, 0] ** 2 + samples[:, 0] * samples[:, 1] + samples[:, 1] ** 2
    statics = {
        'labels mean': labels.mean(),
        'labels std': labels.std() 
    }

    print('statics:', statics) 

    random_shuffle_split_save(samples, labels, 0.9, path=REGRESSION_PATH)


if __name__ == "__main__":
    generate_gaussian()
    generate_regression()