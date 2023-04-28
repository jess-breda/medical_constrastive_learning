import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


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


class TripletsMNIST(Dataset):
    def __init__(self, root="./data", train=True, transform=None, max_triplets=None):
        transform = self.get_default_transform()
        self.mnist_dataset = datasets.MNIST(
            root=root, train=train, transform=transform, download=False
        )
        self.targets = self.mnist_dataset.targets
        self.triplets = self._generate_triplets(_max_triplets=max_triplets)

    def __getitem__(self, index):
        anchor, positive, negative = self.triplets[index]
        anchor_img, anchor_label = self.mnist_dataset[anchor]
        positive_img, _ = self.mnist_dataset[int(positive)]
        negative_img, _ = self.mnist_dataset[int(negative)]
        return anchor_img, positive_img, negative_img

    def get_default_transform(self):
        transform = transforms.Compose(
            [
                transforms.Grayscale(
                    num_output_channels=3
                ),  # Convert to 3 channel grayscale
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        return transform

    def __len__(self):
        return len(self.triplets)

    def _generate_triplets(self, debug=True, _max_triplets=None):
        triplets = []
        N = len(self.mnist_dataset) if _max_triplets is None else _max_triplets
        for i in range(N):
            anchor_img, anchor_label = self.mnist_dataset[i]
            positive_indices = torch.where(self.targets == anchor_label)[0]
            negative_indices = torch.where(self.targets != anchor_label)[0]

            if len(positive_indices) < 2 or len(negative_indices) < 1:
                continue

            positive_idx = positive_indices[
                torch.randint(high=len(positive_indices), size=(1,))
            ]
            negative_idx = negative_indices[
                torch.randint(high=len(negative_indices), size=(1,))
            ]

            triplets.append((i, positive_idx, negative_idx))

            # if _max_triplets is not None:
            #     print(f"WARNING: reduced dataloader size {max_triplets}")
            #     break
        return triplets


def triplet_collate(batch):
    # concatenate anchor, positive, and negative tensors
    anchor, positive, negative = zip(*batch)
    anchor = torch.stack(anchor, dim=0)
    positive = torch.stack(positive, dim=0)
    negative = torch.stack(negative, dim=0)

    # return triplet tensor
    return torch.stack((anchor, positive, negative), dim=0)


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
class SimCLR(nn.Module):
    def __init__(self, verbose=False):
        super(SimCLR, self).__init__()
        # this is f(*) in the paper
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # this is g(*) in the paper
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.verbose = verbose

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.projector(h)
        if self.verbose:
            print("batch size is ", x.shape)
            print("encoded size is ", h.shape)
            print("final size is ", z.shape)
        return h, z


class MnistModel(nn.Module):
    def __init__(self, mode="contrastive"):
        super(MnistModel, self).__init__()

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

        self.mode = mode
        print(f"model initialized in {self.mode} mode")

    def forward(self, x):
        h = self.features(x)
        if self.mode == "constrastive":
            z = self.contrastive_head(h)
        else:
            z = self.classification_head(h)
        return h, z


#############
### LOSS ####
#############


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
