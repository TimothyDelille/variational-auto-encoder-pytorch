import os

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BATCH_SIZE = 100


def round_image_tensor(x):
    return x.round()


def load_train_and_valid_data(batch_size=BATCH_SIZE):
    dataset = datasets.MNIST(
        'data', train=True, download=True, transform=transforms.ToTensor())
    lengths = [int(0.8*len(dataset)), int(0.2*len(dataset))]

    train_dataset, valid_dataset = random_split(dataset, lengths)

    return [
        DataLoader(train_dataset, batch_size=batch_size,
                   shuffle=True, num_workers=os.cpu_count()),
        DataLoader(valid_dataset, batch_size=batch_size,
                   shuffle=False, num_workers=os.cpu_count())
    ]


def load_test_data(batch_size=BATCH_SIZE, shuffle=False):
    dataset = datasets.MNIST(
        'data', train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count())
