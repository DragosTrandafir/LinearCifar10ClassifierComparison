# CIFAR-10 Linear Classifier — From Scratch to PyTorch

Builds a complete image classification pipeline on the CIFAR-10 dataset, starting from bare NumPy and graduating to PyTorch, with systematic experiment tracking via Weights & Biases.

---

## What's Inside

### 1. Softmax — Understanding the Core Building Block
- Implement the softmax function from scratch in NumPy, with a focus on **numerical stability**
- Extend it with a **temperature parameter** and visualize how temperature scales the output probability distribution

### 2. Dataset — CIFAR-10
- Load and preprocess the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (60,000 colour images, 32×32, 10 classes)
- Reshape raw byte arrays into `(H, W, 3)` images and visualize samples

### 3. Linear Classifier from Scratch (NumPy)
- Implement a softmax classifier using only NumPy: weight matrix, bias trick, forward pass
- Train with mini-batch gradient descent and hand-written cross-entropy loss
- Visualize **learned weight templates** — each class's weight row reshaped into a 32×32 image

### 4. Refactor to PyTorch
- Rewrite the same classifier using `nn.Module`, `nn.Linear`, and the Adam optimizer
- Understand how PyTorch's autograd replaces manual gradient computation

### 5. Loss Function Comparison
Train the same model with four different loss functions and compare their behavior:

| Loss Function | Notes |
|---|---|
| RMSE | Regression loss; used here to show why it's wrong for classification |
| Cross-Entropy | Standard classification loss; best performer |
| Multiclass Hinge Loss | Margin-based; Crammer & Singer formulation |
| Label Smoothing | Cross-entropy with soft targets; implemented from scratch |

### 6. Evaluation Metrics
- Implement **confusion matrix**, **accuracy**, **precision**, **recall**, and **F1 score** from scratch using NumPy
- Visualize the confusion matrix as a heatmap

### 7. Hyperparameter Search & Experiment Tracking (W&B)
- Grid search over learning rates and weight decay values
- All runs tracked with [Weights & Biases](https://wandb.ai): training/test loss and accuracy logged per epoch. You can see them here:
  
- https://api.wandb.ai/links/dragostrandafir443-babes/cph75228 - 1EXP_cifar10-linear-learning_rates-reg_strengths
- https://api.wandb.ai/links/dragostrandafir443-babes/v8cx4aod - 1EXP_cifar10-linear-learning_rates_no_reg


- Final summary exported as a `wandb.Table` for side-by-side comparison across all experiments

---

## Key Findings

- **Best loss function:** Cross-entropy (+ label smoothing gives ~1% extra train accuracy)
- **Best learning rate:** `3e-4` — most stable convergence across all experiments
- **Weight decay:** Minimal effect on a single-layer linear model
- Label smoothing helps prevent overconfident predictions but matters less when overfitting is not the primary concern

---

## Tech Stack

- Python, NumPy, Matplotlib
- PyTorch (`nn.Module`, `optim.Adam`, `F.cross_entropy`)
- Weights & Biases (`wandb`) for experiment tracking

---

## How to Run

```bash
# Install dependencies
pip install torch torchvision numpy matplotlib wandb tqdm

# Download dataset (handled inside the notebook)
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

# Open the notebook
jupyter notebook cifar10_linear_classifier.ipynb
```

> A W&B account is required to log experiments. Run `wandb login` before executing the tracking cells.
