# Building, Breaking and Fixing a Neural Network — Fashion-MNIST

**Course:** Deep Learning for Perception
**Institution:** FAST-NUCES
**Semester:** Fall 2026

## Overview

This project builds a feedforward neural network end-to-end on the Fashion-MNIST
dataset, deliberately overfits it, and then systematically repairs it using the
regularisation and tuning methods covered in class. Every design choice is
measured rather than assumed.

The notebook covers all seven required parts:

1. **Backpropagation from scratch** — a two-layer MLP implemented in raw NumPy,
   verified against PyTorch autograd gradients.
2. **Baseline model and activation study** — Sigmoid, Tanh, ReLU, and Leaky ReLU
   compared on validation loss, gradient magnitude, and dead-unit percentage.
3. **Loss functions** — Cross-entropy vs. MSE for classification, plus a small
   MLP trained on a tabular regression dataset (Diabetes).
4. **Optimiser comparison** — SGD, SGD + momentum, RMSProp, and Adam compared
   on convergence speed, final accuracy, and wall-clock time.
5. **Forcing overfitting** — a deliberately oversized network trained on a
   small subset of data to create a large train/validation gap.
6. **Regularisation study** — L2, L1, dropout, batch normalisation, early
   stopping, data augmentation, and more training data, each evaluated against
   the overfitted baseline.
7. **Hyperparameter tuning with 5-fold cross-validation** — random search over
   learning rate, hidden layer width, and dropout rate, followed by a final
   evaluation on the held-out test set.

## Dataset

[Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
— 60,000 training images and 10,000 test images across 10 clothing categories,
each a 28x28 grayscale image flattened to a 784-dimensional vector.

## Repository Structure

```
.
├── DL_ASS01_XXF_YYYY.ipynb   # Main notebook — all 7 parts, executed with outputs
├── README.md                 # This file
└── results_summary.docx      # One-page summary (final score, config, key finding)
```

> Replace `XXF_YYYY` with your actual batch and roll number before submitting.

## How to Reproduce

### 1. Environment

- **Platform:** [Kaggle Notebooks](https://www.kaggle.com/) (recommended, matches
  the environment used to produce these results)
- **Accelerator:** GPU T4 x2
- **Python:** 3.10+ (Kaggle's default environment)

### 2. Dependencies

All dependencies are available by default in the Kaggle Python environment:

```
numpy
pandas
matplotlib
torch
scikit-learn
```

If running locally instead of Kaggle:

```bash
pip install numpy pandas matplotlib torch scikit-learn
```

### 3. Dataset Setup (Kaggle)

1. Open the notebook on Kaggle.
2. Click **Add Input** → search **Fashion MNIST** (Zalando Research) → attach it.
3. Confirm the dataset path by running the first cell (`os.walk('/kaggle/input')`)
   and update `DATA_PATH` in the data-loading cell if the printed path differs.

### 4. Random Seeds

All experiments use `random_state=42` / `torch.manual_seed(42)` /
`np.random.seed(42)` for reproducibility. Re-running the notebook top-to-bottom
should reproduce the reported numbers (minor floating-point variation between
CPU/GPU is expected).

### 5. Running the Notebook

Run all cells in order, top to bottom. Each part depends on variables defined
in earlier cells (`X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`,
`X_full`, `y_full`, `class_names`), so cells should not be run out of order.

> **Note:** Part 6 (Regularisation Study) and Part 7 (Cross-Validation) are
> the most compute-heavy sections. On a T4 GPU these run in a few minutes;
> on CPU, expect significantly longer runtimes.

## Key Results

| Part | Headline Result |
|------|------------------|
| Part 1 | Max gradient difference vs. PyTorch autograd: ~1.98e-08 (W1), ~4.44e-08 (W2) — implementation verified correct |
| Part 2 | ReLU/Leaky ReLU avoid vanishing gradients seen with Sigmoid; ~8.6% dead ReLU units observed |
| Part 5 | Forced overfitting: 99.50% train accuracy vs. 82.76% validation accuracy (16.74% gap) |
| Part 6 | Best gap-reduction-per-accuracy-lost tradeoff: **more training data** (20,000 samples → 11.85% gap, negligible accuracy loss) |
| Part 7 | Tuned model: 89.48% test accuracy vs. 89.41% baseline — a **negligible +0.07 percentage point improvement** |

### The single change that helped most

Adding more training data (scaling from 2,000 → 10,000 → 20,000 samples) gave
the best reduction in the train/validation gap for the smallest cost in
training accuracy, making it the most efficient fix among all regularisation
methods tested.

### Honest note on Part 7

Hyperparameter tuning via random search + 5-fold cross-validation produced a
selected configuration very close to the original Part 2 baseline settings,
resulting in a negligible accuracy improvement (+0.07 percentage points). This
suggests the simple MLP architecture was already near its performance ceiling
on Fashion-MNIST, and further gains would likely require a different model
architecture (e.g. a CNN) rather than further hyperparameter tuning.

## Authors

- Muhammad Zain — 23F-0557
- Rana Muhammad Rajab khan — 23F-0624

## Academic Integrity

This work is submitted in accordance with the course's academic integrity
policy. All code and analysis represent the authors' own work.
