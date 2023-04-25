import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np
import itertools


###################################
#### DATA LOADERS & TRANSFORMS ####
###################################
def get_mnist_data_loaders(batch_size=32):
    # Set up the data transformations
    transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),  # Convert to 3 channel grayscale
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load the MNIST training dataset
    train_set = datasets.MNIST(
        root="./data", train=True, download=False, transform=transform
    )

    # Create a PyTorch DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    # Load the MNIST test dataset
    test_set = datasets.MNIST(
        root="./data", train=False, download=False, transform=transform
    )

    # Create a PyTorch DataLoader for the test set
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False
    )

    return train_set, train_loader, test_set, test_loader


def sub_sample_loaders(loader, n_samples):
    return list(itertools.islice(loader, n_samples))


def contrastive_transforms(inputs):
    """
    Initial transformation function that returns
    two images- one is the identity the second is a random
    crop & resize of the original image

    eventually, this will be made into a flexible function
    with multiple transform options

    params
    ------
    x : tensor (N, C, H, W)
        batch of train images to transform

    returns
    -------
    x_i : tensor (N, C, H, W)
        identity transform of inputs
    x_j : tensor (N, C, H, W)
        ransom resize and crop of inputs
    """

    x_i = inputs
    x_j = torchvision.transforms.functional.resized_crop(
        inputs, top=8, left=8, height=12, width=12, size=(28, 28)
    )

    return x_i, x_j


######################
####    MODELS    ####
######################


class MNIST_Model(pl.LightningModule):
    def __init__(self):
        super(MNIST_Model, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Classification projection head
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1),
        )

        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 128)
        )

    def forward(self, x):
        h = self.features(x)
        class_out = self.classification_head(h)
        contrast_out = self.contrastive_head(h)
        return class_out, contrast_out


### CONSTRASTIVE TRANSFORMS ####

################
### PLOTTING ###
################


def visualize_batch_transform(x_i, x_j):
    """
    Plot for visualing a transform pair of an image
    where x is (Nbacthes, channels, H, W)
    """

    N, C, H, W = x_i.shape
    Nplt = np.min([5, N])

    fig, axarr = plt.subplots(Nplt, 2, figsize=(8, 3 * Nplt))

    for ix in range(Nplt):
        ax1 = axarr[ix, 0]
        ax2 = axarr[ix, 1]

        ax1.imshow(x_i[ix, 0])
        ax2.imshow(x_j[ix, 0])
        if ix == 0:
            ax1.set_title("x_i image")
            ax2.set_title("x_j image")

    return None
