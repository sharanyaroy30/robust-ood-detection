# Robust Out-of-Distribution Detection in Deep Neural Networks

## Overview
Modern neural networks often make highly confident predictions, even when presented with data that differs significantly from their training distribution. This project investigates **out-of-distribution (OOD) detection** methods to improve model reliability and understand failure modes in deep learning systems.

We train a convolutional neural network on the MNIST dataset and evaluate its behavior on both in-distribution data and unseen datasets (e.g., FashionMNIST). Multiple uncertainty estimation techniques are implemented and compared.

---

## Key Features
- Implementation of a CNN for image classification (PyTorch)
- Comparison of multiple OOD detection methods:
  - **Softmax Confidence (Baseline)**
  - **Monte Carlo Dropout (Uncertainty Estimation)**
  - **Energy-Based Scoring (Modern Approach)**
- Visualization of model confidence and uncertainty distributions
- Analysis of model behavior under distribution shift

---

## Methods

### 1. Softmax Confidence
Uses the maximum softmax probability as a confidence score. Lower confidence indicates a higher likelihood of out-of-distribution input.

### 2. Monte Carlo Dropout
Applies dropout at inference time and performs multiple forward passes to estimate predictive uncertainty via variance.

### 3. Energy-Based Detection
Computes an energy score from model logits:
- Lower energy → likely in-distribution  
- Higher energy → likely out-of-distribution  

---

## Results

The following plots show the separation between in-distribution and OOD samples:

- Softmax confidence distributions  
- Monte Carlo dropout uncertainty distributions  
- Energy-based score distributions  

These comparisons highlight how different methods capture model uncertainty under distribution shift.

---

## Tech Stack
- Python
- PyTorch
- NumPy
- Matplotlib

---

## Project Structure
