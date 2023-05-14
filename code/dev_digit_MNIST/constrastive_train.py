### Constrastive training
# Written by Jess Breda edited by Claira Fucetola

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np

from mnist_dev_loading import TripletsMNIST

import sys

sys.path += ["code"]
from models import MnistModel


MODEL_SAVE_PATH = "C:\\Users\\jbred\\github\\medical_constrastive_learning\\models\\"


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
