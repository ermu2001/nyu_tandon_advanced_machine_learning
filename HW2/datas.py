import os.path as osp
import pickle as pkl
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
    trainset.samples = trainset.samples[:round(len(trainset) * 0.1)]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = SyntheticDataset(**data, train=False)
    testset.samples = trainset.samples[:round(len(testset) * 0.1)]
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
