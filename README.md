# Medical Contrastive Learning

Designed, implemented and written by Jess Breda & Claira Fucetola.

COS429 Computer Vision Final Project | Spring 2023

To run on Google Colab, see [here](https://colab.research.google.com/drive/1SurNXBC_GGivpi0qHEhi_qwBMKOS6aIf?usp=sharing)

## Overview 

**Problem**: The emergence of deep learning models has led to significant advances in computer aided medical diagnostics. Yet, despite its importance, this progress has been limited relative to other classification tasks in the field of computer vision [[1](https://arxiv.org/abs/1705.02315)]. The primary cause of this is that high-quality labeled medical data is limited.

**Background**: To address the challenge of limited high-quality labeled data, researchers have explored strategies such as data augmentation, contrastive learning pre-training, and a variety of neural network architectures to improve classification accuracy for medical diagnosis. A successful approach for improving feature extraction in deep learning for chest Xray diagnosis is to pre-train a base network using a *contrastive learning triplet loss function* [[2](https://arxiv.org/abs/1711.05225)]. This approach is less computationally expensive than other contrastive loss functions (e.g. SimCLR, InfoNCE), yet is capable of showing state of the art performance.

**Implementation**: Here, we implemented supervised pre-training with a contrastive triplet loss function and assessed classification performance on the chestMNIST dataset. We examined the effect of supervised pre-training on classification accuracy in a simple convolutional neural network (CNN). Then, we experimented with data augmentation, training data sampling, and modeling complexity. 

**Findings**: In summary we found that pre-training with a contrastive triplet loss on triplets where the positive image was augmented resulted in a *6% increase in accuracy* when compared to no pre-training.

<p align="center">
  <img src="https://github.com/jess-breda/medical_constrastive_learning/assets/53059059/5a97446f-4562-4ecf-80ed-202dc469a3cd" alt="Network Accuracy and Top-1 Accuracy">
</p>

## Details

For a complete write up with all figures and citations, see the [final report](/final_report.pdf).

### Dataset
Here we used the [ChestMNIST dataset](https://medmnist.com/), which consitsts of chest-Xrays with 15 possible labels. Multi-labeled images were removed for our analyses. 

<p align="center">
  <img src="https://github.com/jess-breda/medical_constrastive_learning/assets/53059059/1cd6d159-530e-434b-8339-413a82c7490c" alt="Dataset ChestMNIST" width="75%" height="75%">
</p>

### Model Architecture
The baseline model consist of 3-layer CNN followed by a flattening layer, `h(*)`, for feature extraction with an added classification head, `k(*)`.  The classification head consists of two fully connected layers with ReLU activation, with a dropout layer added in between, and softmax activation on the outputs (left). To improve performance of the network, we added a supervised pre-training step using a contrastive head `z(*)` before the classification head that consisted of two fully connected layers with a ReLU activation (right).

<p align="center">
  <img src="https://github.com/jess-breda/medical_constrastive_learning/assets/53059059/59e4d769-cc12-4432-9fcb-e1f642a24705" alt="Model Arc" width="75%" height="75%">
</p>

### Triplet Loss
To implment learning with the constrastive model we used [triplet margin loss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html). A triplet is a group of anchor (a), positive (p), and negative (n) data points. The anchor and positive data points are of the same class and the negative is of a different class. Supervised pre-training with triplet loss aims to maximize the distance between the embeddings of the anchor and the embeddings of the negative and minimize the distance between the embeddings of the anchor and the embeddings of the positive datapoint.  

<p align="center">
    <img src="https://github.com/jess-breda/medical_constrastive_learning/assets/53059059/13c6a212-77fd-4a1b-9af8-3edcb544a756" alt="Triplet Loss" width="50%" height="50%">
  <img src="https://github.com/jess-breda/medical_constrastive_learning/assets/53059059/fcd20f9c-3834-4270-9fbf-30358490295c" alt="Triplet Loss" width="75%" height="75%">
</p>

### Experiments & Results

**1. Classification with and without contrastive learning pre-training**. In this experiment, we tested whether contrastive learning with randomly selected, non-augmented triplets improves classification performance using a simple CNN. We found there is a 5% increase in overall accuracy. See figures 6-8, 13.

**2. Triplet Modulation** In this experiment, we tested if and how three different types of triplets affected contrastive training. Specifically, we compared embeddings and performance when normal, hard, or augmented triplet sets were used (see Methods- Pre-Training with Contrastive Learning). We found that augmenting positive the triplets led to an additional 1% increase in classification performance. Moreover, when augmentation was applied to the whole train set during fine-tuning of the classification head, there was a 1% increase in top-5 accuracy (overall accuracy stayed the same). See figures 2, 9-11, 13.

**3. Model complexity**. In this experiment, we tested if a deeper, more complex base model `h(*)` (EfficientNet) would improve classification and post contrastive-learning. While the embeddings after constrastive learning appeared more meaningful, the overall classification accuracy signficantly decreased. See figures 5, 12, 13








