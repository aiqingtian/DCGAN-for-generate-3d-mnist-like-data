from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision

# Generate 3D mnist dataset(32*32*32)
class MNIST3D(Dataset):
    def __init__(self):
        self.mnist_train = torchvision.datasets.MNIST('./mnist', train=True, download=True)
        self.mnist_test = torchvision.datasets.MNIST('./mnist', train=False, download=True)
        self.mnist = self.mnist_train + self.mnist_test

    def __len__( self ):
        return len(self.mnist)

    def __getitem__(self, idx):
        img = np.array(self.mnist[idx][0])/255
        img = np.pad(img, (2, 2), 'constant', constant_values=(0, 0))
        img3d = np.zeros((32, 32, 32))
        for j in range(16):
            img3d[j, :, :] = img
        img_torch = torch.from_numpy(img3d).float()
        img_torch = img_torch.view(1, 32, 32, 32)
        return img_torch

