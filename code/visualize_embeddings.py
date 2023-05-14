### TSNE visual of embeddings
# Written by Claira Fucetola

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data_loading import *
from cnn_models import *

# Load Data
print("loading data")
data = np.load(CHEST_MNIST_PATH)
print("Checking data files: ", data.files)
train_dataset, test_dataset, val_dataset = getDataset(data, identity, 4000, False)

labels = LABELS_DICT

train_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=True, pin_memory=True
)

# Get the embeddings before training
with torch.no_grad():
    # Load the model's state_dict from the file
    saved_pretrain_state_dict = torch.load(
        "/content/gdrive/MyDrive/Classes/Spring 2023/COS 429/final_project/checkpoint_JModelAug.pth"
    )

    # Create a new instance of the model and freeze features
    model_pretrain = MnistModel(freeze_features=True).to(device)

    # Load the saved state_dict into the new model
    model_pretrain.load_state_dict(saved_pretrain_state_dict)
    model_pretrain.mode = "constrastive"

    model_pretrain.eval()
    embeddings_before = []
    pre_train_labels = []
    for i, (im, lab) in enumerate(train_loader):
        for row in lab:
            lab = np.where(row == 1)[0]
            if lab.size == 0:
                pre_train_labels.append(0)
            else:
                pre_train_labels.append(lab[0] + 1)

        h, z = model_pretrain(im.to(device))
        embeddings_before.append(h.cpu())
    embeddings_before = torch.cat(embeddings_before, dim=0)

# Get the Embeddings after Training
with torch.no_grad():
    # Load the model's state_dict from the file
    saved_posttrain_state_dict = torch.load(
        "/content/gdrive/MyDrive/Classes/Spring 2023/COS 429/final_project/end_JModelAug.pth"
    )

    # Create a new instance of the model
    model_posttrain = MnistModel(freeze_features=True).to(device)

    # Load the saved state_dict into the new model
    model_posttrain.load_state_dict(saved_posttrain_state_dict)

    model_posttrain.mode = "constrastive"
    model_posttrain.eval()
    embeddings_after = []
    post_train_labels = []
    for i, (im, lab) in enumerate(train_loader):
        for row in lab:
            lab = np.where(row == 1)[0]
            if lab.size == 0:
                post_train_labels.append(0)
            else:
                post_train_labels.append(lab[0] + 1)

        h, z = model_posttrain(im.to(device))
        embeddings_after.append(h.cpu())
    embeddings_after = torch.cat(embeddings_after, dim=0)

# Reduce the dimensionality of the embeddings with t-SNE
tsne = TSNE(n_components=2)
embeddings_before_tsne = tsne.fit_transform(embeddings_before)
embeddings_after_tsne = tsne.fit_transform(embeddings_after)

# Plot the embeddings
cmat = plt.cm.Spectral
norm = plt.Normalize(vmin=0, vmax=14)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(
    embeddings_before_tsne[:, 0],
    embeddings_before_tsne[:, 1],
    c=cmat(norm(pre_train_labels)),
    cmap="tab10",
)
axs[0].set_title("Embeddings before training")
axs[1].scatter(
    embeddings_after_tsne[:, 0],
    embeddings_after_tsne[:, 1],
    c=cmat(norm(post_train_labels)),
    cmap="tab10",
)
axs[1].set_title("Embeddings after training")
plt.show()
