### Data loading for the standard digits MNIST dataset
# Written by Jess Breda

# Note this was specific for debugging purposes where
# the classic MNIST data was used while data loaders for
# the chest MNIST data set were still being written
# see data_loading.py for the proper functions

import torchvision.datasets as datasets
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms
import torch

###################################
#     TRIPLET DATA LOADER         #
###################################


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

        return triplets


def triplet_collate(batch):
    # concatenate anchor, positive, and negative tensors
    anchor, positive, negative = zip(*batch)
    anchor = torch.stack(anchor, dim=0)
    positive = torch.stack(positive, dim=0)
    negative = torch.stack(negative, dim=0)

    # return triplet tensor
    return torch.stack((anchor, positive, negative), dim=0)


###################################
#      STANDARD DATA LOADER       #
###################################
# used with classification only


def get_mnist_data_loaders(
    batch_size=32, shuffle=True, train_subset=None, test_subset=None
):
    # Set up the data transformations
    transform = transforms.Compose(
        [
            transforms.Grayscale(
                num_output_channels=3
            ),  # Convert to 3 channel grayscale
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    # Load the MNIST training dataset
    train_set = datasets.MNIST(
        root="./data", train=True, download=False, transform=transform
    )

    if train_subset:
        # Create a subset of the training dataset with the specified number of datapoints
        train_set = Subset(train_set, range(train_subset))

    # Create a PyTorch DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )

    # Load the MNIST test dataset
    test_set = datasets.MNIST(
        root="./data", train=False, download=False, transform=transform
    )

    if test_subset:
        # Create a subset of the training dataset with the specified number of datapoints
        test_set = Subset(test_set, range(test_subset))

    # Create a PyTorch DataLoader for the test set
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle
    )

    return train_set, train_loader, test_set, test_loader
