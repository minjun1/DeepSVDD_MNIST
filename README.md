# Anomaly Detection in MNIST Dataset with Deep SVDD

This repository provides a PyTorch implementation of the Deep SVDD method for anomaly detection within the MNIST dataset, inspired by the approach presented in the ICML 2018 paper "Deep One-Class Classification" by Lukas Ruff et al.

PDF of the paper is available at: [http://proceedings.mlr.press/v80/ruff18a.html](http://proceedings.mlr.press/v80/ruff18a.html)

## Project Overview

The project demonstrates how to apply the Deep SVDD method for identifying anomalies in digit images from the MNIST dataset. It focuses on training a convolutional encoder model to learn compact representations of "normal" digit images and uses this model to detect anomalous instances.

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Matplotlib
- Numpy
- torchvision

### Installation

1. Clone this repository to your local machine.
2. Ensure you have the required packages installed:
   ```bash
   pip install -r requirements.txt
