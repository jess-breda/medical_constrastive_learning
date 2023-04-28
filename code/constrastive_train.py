import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
import numpy as np
import itertools

"img10k_bs32_ep10_.pth"

###################################
#           DATA LOADER           #
###################################

MODEL_SAVE_PATH = "C:\\Users\\jbred\\github\\med_simCLR\\models\\"


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
#              MODEL              #
###################################


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


###################################
#              TRAIN              #
###################################


def run_contrastive_training(max_triplets, params={}):
    """
    Primary function for running training of contrastive
    model
    """
    # TODO replace with CF code
    print("loading triplets")
    train_set = TripletsMNIST(max_triplets=max_triplets)
    print(f"{len(train_set)} triplets loaded")

    model = MnistModel(mode="constrastive")

    trained_model, loss = contrastive_train(train_set, model, params=params)

    return trained_model, loss


def contrastive_train(train_set, model, params):
    """
    Train function for contrastive triplet loss
    Args:
        model: Object holding the initialized model
        train_set : Containing anchor, anchor label, positive idx, negative idx
        params: Parameters for configuring training
            params["batch_size"]
            params["shuffle_batch"]
            params["learning_rate"]
            params["momentum"]
            params["margin"]
            params["num_epochs"]


        TODO: pass in some sort of test set here as well
    """
    bs = params.get("batch_size", 32)
    sh = params.get("shuffle_batch", True)
    lr = params.get("learning_rate", 0.01)
    mo = params.get("momentum", 0.9)
    mar = params.get("margin", 1.0)
    nep = params.get("num_epochs", 5)
    vb = params.get("verbose", False)
    fname = params.get("fname", "saved_model")

    # TODO get embeddings before training stars
    # H_start, _ = model(train_set_anchor_img) # need to store labels to sort H_start too!
    torch.save(model.state_dict(), (MODEL_SAVE_PATH + fname + "_start.pth"))

    # TODO move train loader into epochs when triplet pairs reshuffling is functional
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=bs, shuffle=sh, collate_fn=triplet_collate
    )

    criterion = nn.TripletMarginLoss(margin=mar, p=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mo)

    loss = -99 * np.ones(nep)
    for ep in range(nep):
        running_loss = 0.0
        epoch_loss = 0.0

        # TODO here is where reshuffling will happen

        for i, (a, p, n) in enumerate(train_loader):
            # forward pass triplets return embedding, projection
            h_a, z_a = model(a)
            h_p, z_p = model(p)
            h_n, z_n = model(n)

            # normalize projections
            z_a = F.normalize(z_a, p=2, dim=1)
            z_p = F.normalize(z_p, p=2, dim=1)
            z_n = F.normalize(z_n, p=2, dim=1)

            # calc loss, backprop, take step
            optimizer.zero_grad()
            batch_loss = criterion(z_a, z_p, z_n)
            batch_loss.backward()
            running_loss += batch_loss.item()
            epoch_loss += batch_loss.item()
            optimizer.step()

            # print statistics
            # running_loss += batch_loss.item()
            if vb:
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f"[{ep + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                    running_loss = 0

        loss[ep] = epoch_loss
        print(f"epoch {ep+1}/{nep} complete, loss: {epoch_loss}")

    print(f"Finished Training with final loss {loss[-1]}")
    torch.save(model.state_dict(), (MODEL_SAVE_PATH + fname + "_end.pth"))

    # TODO get embeddings after training end
    # H_end, _ = model(train_set_anchor_img) # need to store labels to sort H_start too!

    return model, loss  # (h_start, h_end)
