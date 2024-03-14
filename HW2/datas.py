import os.path as osp
import pickle as pkl
import random
from typing import Any, Callable
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

DATA_FOLDER='DATAS'
GAUSSIAN_PATH = osp.join(DATA_FOLDER, 'gaussian.pkl')
REGRESSION_PATH = osp.join(DATA_FOLDER, 'regression.pkl')

class SyntheticDataset(Dataset):
    def __init__(self, train_samples, train_labels, test_samples, test_labels, train=True) -> None:
        super().__init__()
        self.train = train
        if train:
            self.samples = train_samples
            self.labels = train_labels
        else:
            self.samples = test_samples
            self.labels = test_labels
        
    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label

    def __len__(self):
        return self.samples.shape[0]


def get_dataset_2d_regression(batch_size):

    with open(REGRESSION_PATH, 'rb') as f:
        data = pkl.load(f)
    trainset = SyntheticDataset(**data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = SyntheticDataset(**data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = [0, 1, 2, 3]
    return trainset, trainloader, testset, testloader, classes

def get_dataset_2d_gaussian(batch_size):

    with open(GAUSSIAN_PATH, 'rb') as f:
        data = pkl.load(f)
    trainset = SyntheticDataset(**data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = SyntheticDataset(**data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = None
    return trainset, trainloader, testset, testloader, classes



def get_dataset_mnist(batch_size):
    ratio = 0.1 # remaining data as required

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])


    trainset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True,
                                            download=True, transform=transform)
    
    trainset.data = trainset.data[:round(ratio * len(trainset))]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False,
                                        download=True, transform=transform)
    testset.data = trainset.data[:round(ratio * len(trainset))]
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = (str(i) for i in range(1, 11))
    return trainset, trainloader, testset, testloader, classes



def preprocess_get_pair_dataset(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                            shuffle=True, num_workers=2)
    container = dict()
    for org_sample, org_label in loader:
        if org_label.item() not in container:
            container[org_label.item()] = [org_sample]
        else:
            container[org_label.item()].append(org_sample)
    
    container_keys = list(container.keys())
    samples, labels = [], []
    while sum(len(container[k]) for k in container) >= 1:
        try:
            # for a balanced label set.
            is_same = random.random() > 0.5
            if is_same:
                org_label = random.choice(container_keys)
                sample = []
                sample.append(container[org_label].pop())
                sample.append(container[org_label].pop())
                sample = torch.concat(sample, dim=1)
                label = torch.tensor([1])

            else:
                org_label = random.choice(container_keys)
                org_label_diff = random.choice([ i for i in container_keys if i!= org_label])
                sample = []
                sample.append(container[org_label].pop())
                sample.append(container[org_label_diff].pop())
                sample = torch.concat(sample, dim=1)
                label = torch.tensor([0])
        except Exception as e:
            print(e)
        else:
            samples.append(sample)
            labels.append(label)

    samples = torch.concat(samples)
    labels = torch.concat(labels)
    pair_dataset = SyntheticDataset(samples, labels, None, None,)
    return pair_dataset
            


def get_dataset_mnist_pair(batch_size):
    ratio = 0.1 # remaining data as required

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])


    mnist_trainset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=True,
                                            download=True, transform=transform)
    
    mnist_trainset.data = mnist_trainset.data[:round(ratio * len(mnist_trainset))]
    mnist_testset = torchvision.datasets.MNIST(root=DATA_FOLDER, train=False,
                                        download=True, transform=transform)
    mnist_testset.data = mnist_testset.data[:round(ratio * len(mnist_testset))]


    trainset = preprocess_get_pair_dataset(mnist_trainset)
    testset = preprocess_get_pair_dataset(mnist_testset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = (str(i) for i in range(1, 11))
    return trainset, trainloader, testset, testloader, classes




def get_dataset_cifar10(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root=DATA_FOLDER, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_FOLDER, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes 



if __name__ == "__main__":
    trainset, trainloader, testset, testloader, classes = get_dataset_mnist_pair(32)
    from torchvision.transforms.functional import to_pil_image
    import einops
    for data in trainloader:
        print(data)
        print(data[0].shape)
        print(data[1].shape)
        for i, (sample, label) in enumerate(zip(*data)):
            sample = einops.rearrange(sample, 'p h w -> 1 h (p w)')
            img = to_pil_image(sample)
            img.save(f'tmp/{i}_{label}.png')
        break
