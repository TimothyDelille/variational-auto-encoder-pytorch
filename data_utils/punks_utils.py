import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BATCH_SIZE = 100


class PunksDataset(Dataset):
    def __init__(self, path="./data/punks.png", transform=transforms.ToTensor(), start_idx=0, stop_idx=9999):
        self.punks = np.array(Image.open(path).convert('RGB'))
        # 10,000 crypto punks, 24x24 each
        self.transform = transform
        self.start_idx = start_idx
        self.stop_idx = stop_idx

    def __getitem__(self, idx):
        # 100 punks per row
        idx = self.start_idx + idx
        i = int((idx//100) * 24)
        j = int(idx % 100 * 24)

        img = self.punks[i:i+24, j:j+24, :]

        if self.transform:
            img = self.transform(img)

        return img, idx  # img, label

    def __len__(self):
        return self.stop_idx - self.start_idx + 1


def load_train_and_valid_data(batch_size=BATCH_SIZE):
    train_dataset = PunksDataset(start_idx=0, stop_idx=5999)
    valid_dataset = PunksDataset(start_idx=6000, stop_idx=7999)

    return [
        DataLoader(train_dataset, batch_size=batch_size,
                   shuffle=True, num_workers=os.cpu_count()),
        DataLoader(valid_dataset, batch_size=batch_size,
                   shuffle=False, num_workers=os.cpu_count())
    ]


def load_test_data(batch_size=BATCH_SIZE, shuffle=False):
    test_dataset = PunksDataset(start_idx=8000, stop_idx=9999)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
