# Skin-Disease-Image-Classification

## Project Overview

This project implements an end-to-end deep learning system for automated classification of dermatological conditions from clinical images. The pipeline covers the full lifecycle of a modern machine learning project: data ingestion, preprocessing and augmentation, model training and fine-tuning using transfer learning, systematic evaluation, and ensemble-based inference.

The final model predictions were submitted to a Kaggle competition hosted by MIT.
**Our team secured 3rd place out of 74 teams**, achieving a **final weighted F1 score of 0.71**.

### Competition Participation Statistics

* 421 Entrants
* 351 Participants
* 74 Teams
* 670 Submissions

---

## Dataset Description

The dataset is a curated subset of the **FitzPatrick17k** dermatology dataset. It contains approximately **4,500 labeled images** spanning **21 distinct skin conditions** selected from over 100 diagnostic categories in the original dataset. The subset preserves clinically meaningful class diversity while keeping the problem computationally manageable.

The images represent both serious medical conditions (e.g., melanoma) and cosmetic conditions (e.g., acne) across a broad spectrum of skin tones scored on the FitzPatrick Skin Tone (FST) scale. This introduces realistic challenges related to representation, class imbalance, and clinical variability.

### Data Structure

```
bttai-ajl-2025/
│
├── train.csv                 # Image metadata and ground-truth labels
├── test.csv                  # Metadata for unlabeled test images
│
├── train/
│   └── train/
│       ├── <label>/
│       │   └── <md5hash>.jpg
│       └── ...
│
├── test/
│   └── test/
│       └── <md5hash>.jpg
│
└── augmented_train/          # Optional offline augmented images
    └── <md5hash>.jpg
```

Each image is identified by an `md5hash`. The training labels correspond to one of 21 dermatological diagnoses.

---

## Model Architecture and Training Strategy

### Neural Network Models

The pipeline trains and fine-tunes three modern convolutional neural networks initialized with ImageNet pretrained weights:

* **ResNet18** — primary training model
* **EfficientNet-B0** — ensemble model
* **DenseNet121** — ensemble model

The final classification layers of each network are replaced to match the 21 target classes.

### Training Strategy

* **Transfer learning** with pretrained backbones
* **Fine-tuning** of all network parameters
* **Stratified K-Fold cross-validation** for robust train/validation splitting
* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** Adam
* **Learning Rate Scheduler:** ReduceLROnPlateau
* **Input Resolution:** 224 × 224
* **Batch Size:** 32
* **Ensemble inference** via averaged logits from all trained models

The training process logs loss and accuracy for both training and validation splits and produces full evaluation artifacts including learning curves, classification reports, and confusion matrices.

---

## Data Augmentation and Preprocessing

To improve generalization and reduce overfitting, the following augmentations are applied during training:

* Image resizing to 224 × 224
* Random horizontal flipping
* Random rotation (±20 degrees)
* Normalization using ImageNet mean and standard deviation

Validation and test data undergo deterministic preprocessing without random augmentation.

---

## Inference and Competition Results

Final predictions are produced using an **ensemble of ResNet18, EfficientNet-B0, and DenseNet121**, with model logits averaged before classification.

Predictions are exported in the required Kaggle format:

```
md5hash,label
```

### Final Performance

* **Competition Rank:** 3rd out of 74 teams
* **Final Weighted F1 Score:** 0.71

This result demonstrates the effectiveness of the model architecture, training strategy, and ensemble design on a clinically realistic and challenging dataset.
