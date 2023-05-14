### Constrastive leanring pre training
# Written by Jess Breda & Claira Fucetola


import torch
from data_loading import *
import torch.nn as nn
import torch.nn.functional as F
from cnn_models import *


###################################
#              TRAIN              #
###################################


def run_contrastive_training(max_triplets, labels=LABELS_DICT, params={}):
    """
    Primary function for running training of contrastive
    model
    """

    print("loading data")
    data = np.load(CHEST_MNIST_PATH)
    print("Checking data files: ", data.files)
    train_dataset, test_dataset, val_dataset = getDataset(
        data, identity, max_triplets, False
    )

    ## TRAIN ##
    # get locations of positive and negative indicies for each anchor label
    print("\nloading train triplet indicies")
    train_triples_idx = Triplets(train_dataset, labels)
    print(f" Train {len(train_triples_idx)} triplet indices")

    # load dataset anchor image (current image in dataset), positive image
    # (chosen from the list associated with anchors label), negative image
    print("loading train triplets")

    # NOTE: change transforms = transform when doing aug
    train_triplets = updateTriplet(
        train_dataset, train_triples_idx, labels, transforms=transform
    )
    print(f"{len(train_triplets)} train triplets")

    ## VAL ##
    # get locations of positive and negative indicies for each anchor label
    print("\n loading val triplet indicies")
    val_triples_idx = Triplets(val_dataset, labels)
    print(f" Train {len(val_triples_idx)} val indices")

    # load dataset anchor image (current image in dataset), positive image
    # (chosen from the list associated with anchors label), negative image
    print("loading triplets")
    val_triplets = updateTriplet(val_dataset, val_triples_idx, labels)
    print(f"{len(val_triplets)} val triplets\n")

    model = MnistModel(mode="constrastive").to(device)

    data_dict = {
        "train_dataset": train_dataset,
        "train_triples_idx": train_triples_idx,
        "train_triplets": train_triplets,
        "val_dataset": val_dataset,
        "val_triples_idx": val_triples_idx,
        "val_triplets": val_triplets,
        "labels": labels,
    }

    trained_model, train_loss, val_loss = contrastive_train(
        model, data=data_dict, params=params
    )

    return trained_model, train_loss, val_loss


def contrastive_train(model, data, params):
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
    """
    bs = params.get("batch_size", 32)
    sh = params.get("shuffle_batch", True)
    lr = params.get("learning_rate", 0.01)
    mo = params.get("momentum", 0.9)
    mar = params.get("margin", 1.0)
    nep = params.get("num_epochs", 5)
    vb = params.get("verbose", False)
    vf = params.get("val_freq", 5)

    fname = params.get("fname", "saved_model")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # initalize train_loader from triplets sent in
    # NOTE: change to data[train_dataset] when doing hard trip
    train_loader = torch.utils.data.DataLoader(
        data["train_triplets"], batch_size=bs, shuffle=sh, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        data["val_triplets"],
        batch_size=bs,
        shuffle=sh,
        collate_fn=triplet_collate,
        num_workers=0,
    )

    criterion = nn.TripletMarginLoss(margin=mar, p=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mo)

    train_loss = -99 * np.ones(nep)
    val_loss = 99 * np.ones(int(nep / vf))

    # variables for early stopping
    best_val_loss = float("inf")
    patience = 5
    counter = 0
    for ep in range(nep):
        running_loss = 0.0
        train_epoch_loss = 0.0

        # get a new triplets each epoch and reload into dataloader
        # NOTE: Comment this out when doing Hard Trips
        # NOTE: when doing aug, transforms = transform
        shuffled_train_triplets = updateTriplet(
            data["train_dataset"],
            data["train_triples_idx"],
            data["labels"],
            transforms=transform,
        )
        train_loader = torch.utils.data.DataLoader(
            shuffled_train_triplets,
            batch_size=bs,
            shuffle=sh,
            collate_fn=triplet_collate,
        )

        ### TRAIN ###

        # For hard trip i, (im, target)
        # For normal i, (a, p, n)
        for i, (a, p, n) in enumerate(train_loader):
            # forward pass triplets return embedding, projection
            ######################################
            ###         HARD TRIPLETS          ###
            # h, z = model(im.to(device))

            # targ_labels = []

            # For hard trip only
            # for row in target:
            # lab = np.where(row == 1)[0]
            # if lab.size == 0:
            # targ_labels.append(0)
            # else:
            # targ_labels.append(lab[0]+1)

            # target = torch.tensor(targ_labels)

            #########################################

            # for normal trip
            h_a, z_a = model(a.to(device))
            h_p, z_p = model(p.to(device))
            h_n, z_n = model(n.to(device))

            # normalize projections
            z_a = F.normalize(z_a, p=2, dim=1)
            z_p = F.normalize(z_p, p=2, dim=1)
            z_n = F.normalize(z_n, p=2, dim=1)

            # calc loss, backprop, take step
            optimizer.zero_grad()

            # normal trip
            batch_loss = criterion(z_a, z_p, z_n)

            # batch_loss = batch_hard_triplet_loss(target.to(device), h, margin=mar)

            batch_loss.backward()
            running_loss += batch_loss.item()
            train_epoch_loss += batch_loss.item()
            optimizer.step()

            # print statistics
            # running_loss += batch_loss.item()
            if vb:
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f"[{ep + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
                    running_loss = 0

        train_loss[ep] = train_epoch_loss / len(train_loader)
        print(
            f"epoch {ep+1}/{nep} complete, loss: {train_epoch_loss / len(train_loader)}"
        )

        ### VALIDATION ###
        ## ASSUMES YOU ARE PASSING THE WHOLE VAL SET IN SINGLE BATCH##
        if (ep + 1) % vf == 0:
            # print("entering validation logic")

            with torch.no_grad():
                # print("passing through with statement")
                val_running_loss = 0.0
                model.eval()

                ### ISSSUE IS HERE! LEN of val_loader is 1
                # print(f"LEN VAL LOADER {len(val_loader)}")
                for i, (a, p, n) in enumerate(val_loader):
                    # print(f"\n iteration: {i, a.size(), p.size(), n.size()}")
                    h_a, z_a = model(a.to(device))
                    h_p, z_p = model(p.to(device))
                    h_n, z_n = model(n.to(device))

                    z_a = F.normalize(z_a, p=2, dim=1)
                    z_p = F.normalize(z_p, p=2, dim=1)
                    z_n = F.normalize(z_n, p=2, dim=1)

                    # Calculate loss (ASSUME SINGLE BATCH!!)
                    idx = int(((ep + 1) / vf) - 1)
                    val_loss[idx] = criterion(z_a, z_p, z_n).item()

            print(f"Validation loss: {val_loss[idx]:.3f}")
            model.train()

        if ep == 50:
            torch.save(
                model.state_dict(),
                (MODEL_SAVE_PATH + "JModelAug_50.pth"),
            )
        if ep == 100:
            torch.save(
                model.state_dict(),
                (MODEL_SAVE_PATH + "JModelAug_1-0.pth"),
            )

        if ep == 150:
            torch.save(
                model.state_dict(),
                (MODEL_SAVE_PATH + "JModelAug_150.pth"),
            )

    print(f"Finished Training!")
    torch.save(
        model.state_dict(),
        (MODEL_SAVE_PATH + "end_JModelAug.pth"),
    )

    return model, train_loss, val_loss
