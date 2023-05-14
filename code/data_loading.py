### Data Loading specific for chest MedMnist
# Written by Claira Fucetola edited by Jess Breda

import torch
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, Subset, ConcatDataset
import itertools

###################################
#            GLOBALS              #
###################################

CHEST_MNIST_PATH = 'C:\Users\jbred\github\medical_constrastive_learning\code\data\CHESTMNIST\chestmnist.npz'

LABELS_DICT = {
    "hernia": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "pleural": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "fibrosis": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "emphysema": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "edema": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "consolidation": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "pneumothorax": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "pneumonia": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "nodule": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "mass": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "infiltration": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "effusion": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "cardiomegaly": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "atelectasis": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "normal": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

###################################
#           TRANSFORMS            #
###################################

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((300, 300)),
        transforms.RandomCrop((250, 250)),
        transforms.RandomResizedCrop(32),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

identity = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(
            (32, 32)
        ),  # we can change this to be whatever size of image we want
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]
)

#################################### CONSTRASTIVE ########################################

###################################
#             DATASET             #
###################################

class ChestDataset(Dataset):
    def __init__(self, data, labels, max = None, transforms=None):
        self.data = data
        self.labels = labels
        self.transforms = transforms
        self.indices = []
        N = len(self.data) if max is None else max
        count = 0
        for i in range(N):

            if np.sum(self.labels[i]) > 1 :
                continue
            else:
                self.indices.append(i)
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, idx):
        x = self.data[self.indices[idx]]
        y = self.labels[self.indices[idx]]


        if self.transforms is not None:
          x = self.transforms(x)
        return x, y
    
def getDataset(data, identity, max, split):
  # train data
  train_data = data['train_images']
  train_lab = data['train_labels']

  #test data
  test_data = data['test_images']
  test_lab = data['test_labels']

  #val data
  val_data = data['val_images']
  val_lab = data['val_labels']

  #original dataset (for test and val)
  train_dataset = ChestDataset(train_data, train_lab, max, transforms = identity)
  test_dataset = ChestDataset(test_data, test_lab, max, transforms = identity)
  val_dataset = ChestDataset(val_data, val_lab, max, transforms = identity)

  return train_dataset, test_dataset, val_dataset


###################################
#            TRIPLETS             #
###################################

class updateTriplet(Dataset):
    """
    used on an epoch basis to get a new subset of anchor, positive
    and negative triplets
    """
    def __init__(self, dataset, triplet, labels, transforms=None):
        self.triplet = triplet #dictonary of anchors and positive and negative indices
        self.dataset = dataset
        self.transforms = transforms
        self.labels = labels
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        anchor_im, anchor_lab = self.dataset[idx]
        pos, neg = get_triplet(self.dataset, self.triplet, self.labels, anchor_lab)

        # seperate images and labels for each use in and with dataloader
        pos_im, pos_lab = pos
        neg_im, neg_lab = neg
        anchor = (anchor_im, anchor_lab)
        if self.transforms is not None:
          pos_im = self.transforms(pos_im)

        # might want to change what this returns so that it is easy to understand
        # from the dataloader pov
        return anchor_im, pos_im, neg_im
    

def Triplets(dataset, dic, train=True):
  """
  return a dictionary of anchors and lists of positive and negative indices 
  given a dataset
  """
  triplets = {}
  labels = []
  for i in range(len(dataset)):
    _, label = dataset[i]
    labels.append(label)
  for i in range(len(dataset)):      
        anchor_img, anchor_label = dataset[i]
        lab = list(filter(lambda x: (dic[x] == anchor_label).all(), dic))[0]

        if lab not in triplets:
            positive_indices = np.where((labels == anchor_label).all(axis=1))[0]
            
            negative_indices = np.where((labels != anchor_label).any(axis=1))[0] # finds where any index doesnt match
   
            
            if len(positive_indices) < 2 or len(negative_indices) < 1:
              continue

            positive_indices = itertools.cycle(positive_indices)
            negative_indices = itertools.cycle(negative_indices)
            triplets.update({lab: (positive_indices, negative_indices)})

        if len(triplets) == 15:
          print("all 15 class labels found")
          break

  return triplets

def get_triplet(dataset, triplet, label_dic, label):
  
  """
  get a random index of a positive image and a negative image 
  given a triplet of anchor, list in indices positive, list of indices negative
  where dataset is the orginal dataset
  """
  # get positive and negative indices from label
  lab = list(filter(lambda x: (label_dic[x] == label).all(), label_dic))[0]
  #print(lab)
  positive_indices, negative_indices = triplet[lab]

  # get the next positive image
  positive_idx = next(positive_indices)
  positive_img, positive_label = dataset[positive_idx]

  # get the next negative image
  negative_idx = next(negative_indices)
  negative_img, negative_label = dataset[negative_idx]

  return (positive_img, positive_label), (negative_img, negative_label)


def triplet_collate(batch):
    """
    Quick function to make triplet dataloader work
    """
    # concatenate anchor, positive, and negative tensors
    anchor, positive, negative = zip(*batch) # the way this is written, current anchor, positive, and negative have labels with them but can 
    anchor = torch.stack(anchor, dim=0)
    positive = torch.stack(positive, dim=0)
    negative = torch.stack(negative, dim=0)

    # return triplet tensor
    return torch.stack((anchor, positive, negative), dim=0)

##########

def balance_data(dataset):
  """
  Used to balance the chest MNIST dataset due to 
  a large proportion of "normal" chest labels
  """
  labels =[]
  for i in range(len(dataset)):
    lab = np.where(dataset[i][1] == 1)[0]
    if lab.size == 0:
        labels.append(0)
    else:
        labels.append(lab[0]+1)

  labels_unique, counts = np.unique(labels, axis =0, return_counts = True)
  class_weights = [sum(counts)/c for c in counts]
  print(class_weights)
  print(labels)
  example_weights = [class_weights[e] for e in labels]
  sampler = WeightedRandomSampler(example_weights, len(labels))

  return sampler


################################### CLASSIFICATION ##########################################

def get_mnist_data_loaders(
    batch_size=32, shuffle=True, train_subset=None, test_subset=None
):

    # load in data
    data = np.load(CHEST_MNIST_PATH)
    # Load the ChestMNIST training dataset
    train_set, test_set, val_set = getDataset(data, identity, max = None, split = False)
    
    #################################################
    ##       BUILDING DATASET WITH AUGMENTATION    ##
    #aug_dataset = augmentData(data, transform)
    #train_set = ConcatDataset([train_set, aug_dataset])
    ##################################################

    if train_subset:
        # Create a subset of the training dataset with the specified number of datapoints
        train_set = Subset(train_set, range(train_subset))
        

  # Balance dataset
    train_sampler = balance_data(train_set)

  # Create a PyTorch DataLoader for the training set
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler = train_sampler
    )


    if test_subset:
        # Create a subset of the validation dataset with the specified number of datapoints
        val_set = Subset(val_set, range(test_subset))

    # Balance dataset
    val_sampler = balance_data(val_set)

    # Create a PyTorch DataLoader for the training set
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, sampler=val_sampler)

    return train_set, train_loader, val_set, val_loader

def augmentData(data, transform):
  train_data = data['train_images']
  train_lab = data['train_labels']

  # augment the dataset according to transformations
  aug_dataset = ChestDataset(train_data, train_lab, transforms = transform)

  return aug_dataset