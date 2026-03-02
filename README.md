# Deep Learning Assignment

**Author:** Zunaira Zaheer  
**Roll No:** MSDS25049  
**Date:** 16-02-2025  

---

## 📌 Overview

This repository contains two supervised learning implementations:

1. **Linear Regression (from scratch using NumPy)**
2. **Logistic Regression (using PyTorch)**

The goal of this assignment was to understand gradient-based optimization, normalization, hyperparameter tuning, and model stability through hands-on implementation.

---

# 📊 Task 1 — Linear Regression (NumPy Implementation)

## 🎯 Objective

Predict median house values using 8 numerical features from the **California Housing dataset**.

---

## 🛠 Implementation Details

- Implemented completely from scratch (no ML libraries)
- Manual:
  - Forward propagation
  - L2 loss computation
  - Backpropagation (analytical gradients)
  - Mini-batch SGD updates
- 85%–15% train-validation split
- Z-score normalization (computed from training data only)
- He weight initialization
- Gradient clipping
- Batch size: 64
- Epochs: 100

---

## 🏗 Model Architecture

Note: No nonlinear activation was used, so the overall mapping remains linear.

---

## 🔬 Hyperparameters Explored

- Learning Rates: `0.001`, `0.0005`
- Hidden Units: `2`, `8`
- With and Without Normalization
- 100 Training Epochs

---

## 📈 Results Summary

### 🚨 Without Normalization
- Unstable training
- Very high validation loss
- Negative R² (worse than predicting the mean)

### ✅ With Normalization
- Smooth convergence
- Validation Loss ≈ 0.25–0.28
- Validation R² ≈ 0.60

---

## 🏆 Final Selected Model

- Normalization: ✅
- Hidden Size: 2
- Learning Rate: 0.001

**Performance:**
- Validation Loss ≈ 0.26
- Validation R² ≈ 0.60

---

## 📊 Approximate Comparison Table

| Configuration | Val Loss | Val R² | Outcome |
|--------------|----------|--------|---------|
| NonNorm | H=2 | LR=0.001 | ~20 | ~ -30 | Unstable |
| NonNorm | H=8 | LR=0.001 | ~910 | ~ -1400 | Divergent |
| Norm | H=2 | LR=0.001 | ~0.26 | ~0.60 | ✅ Best |
| Norm | H=8 | LR=0.001 | ~0.25 | ~0.60 | Stable |

---

# 🤖 Task 2 — Logistic Regression (PyTorch)

## 🎯 Objective

Binary classification using logistic regression.

---

## 🛠 Implementation Details

- Implemented in PyTorch
- Linear layer + Sigmoid activation
- Cross-Entropy Loss
- Mini-batch training
- Feature normalization
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 📉 Performance Comparison

### ❌ Without Normalization

- Loss remained high (7–10)
- Unstable training
- High false positives

**Metrics:**
- Accuracy ≈ 0.681
- Precision ≈ 0.542
- Recall ≈ 0.796
- F1 ≈ 0.645

---

### ✅ With Normalization

- Smooth convergence
- Balanced classifier
- Reduced false positives

**Metrics:**
- Accuracy ≈ 0.763
- Precision ≈ 0.667
- Recall ≈ 0.694
- F1 ≈ 0.680

Normalization improved accuracy by ~8%.

---

## 📊 Confusion Matrix Comparison

| Run | TN | FP | FN | TP |
|-----|----|----|----|----|
| Non-Normalized | 53 | 33 | 10 | 39 |
| Normalized | 69 | 17 | 15 | 34 |

---

# 📁 Repository Structure

---

# 🧠 Key Takeaways

- Feature normalization is critical for gradient-based learning.
- Learning rate controls convergence speed and stability.
- Increasing model complexity does not help without proper preprocessing.
- Validation loss should guide model selection.
- Proper initialization improves training efficiency.

---

# 🚀 How to Use Saved Models

### Load Task 2 Model (PyTorch)

```python
import torch

model = torch.load("model_task2.pkl")
model.eval()
This project demonstrates the importance of preprocessing, optimization stability, and hyperparameter tuning in both regression and classification tasks. Proper normalization transformed unstable training into reliable convergence across both tasks.
