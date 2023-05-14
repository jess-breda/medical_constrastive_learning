### Top 1, Top 5, AUC metrics
# Written by Claira Fucetola

import torch
from torchvision import transforms, datasets


from torch.utils.data import DataLoader
from data_loading import CHEST_MNIST_PATH, getDataset, identity, balance_data
from cnn_models import MnistModel

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the model
saved_posttrain_state_dict = torch.load(
    "/content/gdrive/MyDrive/Classes/Spring 2023/COS 429/final_project/end_classificationHardTrip.pth"
)
model = MnistModel(mode="classification", freeze_features=True)
# Load the saved state_dict into the new model
model.load_state_dict(saved_posttrain_state_dict)

# load in data
data = np.load(CHEST_MNIST_PATH)
# Load the ChestMNIST training dataset
train_set, test_set, val_set = getDataset(data, identity, max=None, split=False)
# Balance dataset
test_sampler = balance_data(test_set)

# Create a PyTorch DataLoader for the training set
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, sampler=test_sampler)


# Evaluate the model on the test dataset
model.eval()
with torch.no_grad():
    all_probs = []
    all_labels = []
    for inputs, labels in test_loader:
        _, outputs = model(inputs)
        # Convert labels to 0-14
        # pred = outputs.argmax(dim=0)
        targ_labels = []
        for row in labels:
            lab = np.where(row == 1)[0]
            if lab.size == 0:
                targ_labels.append(0)
            else:
                targ_labels.append(lab[0] + 1)

        targ_labels = torch.tensor(targ_labels)
        all_probs.append(outputs)
        all_labels.append(targ_labels)

# Concatenate the predicted probabilities for all batches
all_probs = torch.cat(all_probs, dim=0)
all_labels = torch.cat(all_labels, dim=0)


# Calculate the AUC for each class
auc_scores = []
print(all_probs.shape)
print(all_labels.shape)

# Reshape the input arrays
# all_probs = all_probs.reshape(-1, 1)
# all_labels= all_labels.reshape(-1, 1)

for i in range(15):
    auc = roc_auc_score((all_labels == i).numpy(), all_probs[:, i].numpy())
    auc_scores.append(auc)

for i, auc in enumerate(auc_scores):
    print(f"Class {i}: AUC = {auc:.4f}")


### AUC Plot ###
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(all_probs.shape[1]):
    fpr[i], tpr[i], _ = roc_curve((all_labels == i).numpy(), all_probs[:, i].numpy())
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
for i in range(all_probs.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Classification With Pre-Training (Hard Triplets")
plt.legend(loc="lower right")
plt.show()


### TOP 1/5 ###
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0
top1_correct = 0
top5_correct = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        _, outputs = model(images)
        targ_labels = []
        for row in labels:
            lab = np.where(row == 1)[0]
            if lab.size == 0:
                targ_labels.append(0)
            else:
                targ_labels.append(lab[0] + 1)

        targ_labels = torch.tensor(targ_labels)
        _, predicted = torch.max(outputs, 1)
        total += targ_labels.size(0)
        correct += (predicted == targ_labels).sum().item()
        _, top5_predicted = torch.topk(outputs, 5, dim=1)
        top1_correct += (predicted == targ_labels).sum().item()
        top5_correct += sum(
            [targ_labels[i] in top5_predicted[i] for i in range(targ_labels.size(0))]
        )

print("Accuracy of the network on the test images: %d %%" % (100 * correct / total))
print(
    "Top-1 Accuracy of the network on the test images: %d %%"
    % (100 * top1_correct / total)
)
print(
    "Top-5 Accuracy of the network on the test images: %d %%"
    % (100 * top5_correct / total)
)
