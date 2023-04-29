import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constrastive_train import MnistModel, MODEL_SAVE_PATH


###################################
#           DATA LOADER           #
###################################


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


def run_classification_training(params):
    """
    params: Parameters for configuring training
        params["train_subset]
        params["test_subset]
        params["batch_size"]
        params["shuffle_batch"]
        params["learning_rate"]
        params["momentum"]
        params["num_epochs"]
        params["fname"]
    """

    train_subset = params.get("train_subset", 10000)
    test_subset = params.get("test_subset", 2000)
    bs = params.get("batch_size", 32)
    sh = params.get("shuffle_batch", True)

    _, train_loader, _, test_loader = get_mnist_data_loaders(
        batch_size=bs, train_subset=train_subset, test_subset=test_subset, shuffle=sh
    )
    print(
        f"{len(train_loader.dataset)} train images and {len(test_loader.dataset)} test images loaded"
    )

    model = MnistModel(mode="classification")

    trained_model, performance = classification_train(
        train_loader, test_loader, model, params=params
    )

    return trained_model, performance


def classification_train(train_loader, test_loader, model, params):
    # get params
    lr = params.get("learning_rate", 0.01)
    mo = params.get("momentum", 0.9)
    nep = params.get("num_epochs", 5)
    vf = params.get("val_freq", 5)
    fname = params.get("fname", "classification_model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mo)

    epoch_train_loss = -99 * np.ones(nep)
    epoch_train_acc = -99 * np.ones(nep)
    epoch_val_loss = -99 * np.ones(int(nep / vf))
    epoch_val_acc = -99 * np.ones(int(nep / vf))

    pre_train_conf_mat = generate_confusion_matrix(model, test_loader)

    ## TRAINING LOOP
    for ep in range(nep):
        # initialize
        train_loss = 0.0
        train_correct = 0

        for data, target in train_loader:
            model.train()  # turn on batch norm layers
            optimizer.zero_grad()  # reset gradients
            _, z = model(data)  # embedding, softmax
            loss = criterion(z, target)  # compute loss
            loss.backward()  # calculate gradients
            optimizer.step()  # update weights

            # get predicted label
            pred = z.argmax(dim=1, keepdim=True)

            # update counters
            train_loss += loss.item() * data.size(0)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

        # calculate train summary
        train_loss /= len(train_loader.dataset)  # batch averaged (not summed)
        train_acc = 100.0 * train_correct / len(train_loader.dataset)
        epoch_train_loss[ep] = train_loss
        epoch_train_acc[ep] = train_acc

        print(
            f"Epoch {ep + 1} / {nep} complete, train loss: {train_loss:.6f}, acc: {train_acc:.2f}%"
        )

        # run test
        if (ep + 1) % vf == 0:
            val_loss, val_acc = run_classification_validation(
                test_loader, model, criterion
            )

            # update counters
            idx = int(((ep + 1) / vf) - 1)
            epoch_val_loss[idx] = val_loss
            epoch_val_acc[idx] = val_acc

    print(f"Finished Training!")
    post_train_conf_mat = generate_confusion_matrix(model, test_loader)

    performance = {
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc,
        "pre_train_conf_mat": pre_train_conf_mat,
        "post_train_conf_mat": post_train_conf_mat,
    }
    torch.save(model.state_dict(), (MODEL_SAVE_PATH + fname + ".pth"))

    return model, performance


# def classification_train(train_loader, test_loader, model, params):
#     lr = params.get("learning_rate", 0.01)
#     mo = params.get("momentum", 0.9)
#     nep = params.get("num_epochs", 5)
#     fname = params.get("fname", "classification_model")

#     # compute confusion matrix pre_training


###################################
#              EVAL               #
###################################


def run_classification_validation(test_loader, model, criterion):
    # initialize
    val_loss = 0.0
    val_correct = 0
    model.eval()  # turn off batch norm layers

    # make sure gradients aren't be calculated
    with torch.no_grad():
        for data, target in test_loader:
            # forward pass full test set
            _, z = model(data)

            # calculate loss
            loss = criterion(z, target)

            # get predicted label
            pred = z.argmax(dim=1, keepdim=True)

            # update counters
            val_loss += loss.item() * data.size(0)
            val_correct += pred.eq(target.view_as(pred)).sum().item()

    # calculate validation summary
    val_loss /= len(test_loader.dataset)
    val_acc = 100.0 * val_correct / len(test_loader.dataset)
    print(f"Val Loss: {val_loss:.6f}, Accuracy: {val_acc:.2f}%")

    # TODO- return if you want!
    return val_loss, val_acc


def generate_confusion_matrix(model, test_loader):
    # Set your model to evaluation mode
    model.eval()

    # Disable gradient calculations using torch.no_grad()
    with torch.no_grad():
        # Loop over the test data and perform inference
        y_true = []
        y_hat = []

        for data, target in test_loader:
            # Forward pass
            _, z = model(data)

            # Compute the predictions
            pred = z.argmax(dim=1)

            # Append the true and predicted labels to the lists
            y_true.append(target.numpy())
            y_hat.append(pred.numpy())

        # Concatenate the lists into arrays
        y_true = np.concatenate(y_true)
        y_hat = np.concatenate(y_hat)

        # Compute the confusion matrix
        conf_mat = confusion_matrix(y_true, y_hat)

    return conf_mat


def plot_confusion_matrix(conf_mat, ax, title="Confusion Matrix"):
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d", ax=ax)

    # Set the axis labels and title
    ax.set(xlabel="Predicted Label", ylabel="True Label", title=title)

    return None
