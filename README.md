# Pytorch Implementation of MD4: Simplified and Generalized Masked Diffusion for Discrete Data

This repository provides a **simple and straightforward** implementation of **MD4**, a generative model based on the paper [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329).

## Overview

This code implements MD4, a discrete diffusion process, that can be used with any discrete dataset. The `MD4Generation` class is designed to support the MD4 generative process with respect to any dataset, but the rest of the script specializes to **Binary MNIST**, where the only values present are:

- `0`
- `1`
- `2` (mask token)

For the neural network architecture, we use a U-Net similar to the one in [Improved Diffusion](https://github.com/openai/improved-diffusion). Specifically, the network consists of:

- A **3x32 embedding layer**, mapping the three possible values (0, 1, and mask token 2) to 32 values, stacked on the channel dimension.
- A subsequent **U-Net** that processes the 32-channel input.
- An output of shape **1x32x32**, after which a **sigmoid activation** would determine the probability of observing a `1` (the network predicts the initial state from a noisy state)

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

The main script is `pytorch_md4.py`, which supports two modes: **training** and **generation**.

### Training

Train the MD4 model on Binary MNIST with basic data augmentation (random horizontal flipping):

```bash
python pytorch_md4.py train --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 5e-4
```

#### Arguments:

- `--epochs`: Total number of training epochs.
- `--checkpoint_interval`: Save a checkpoint every C epochs.
- `--batch_size`: Batch size for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--checkpoint_dir`: *(Optional)* Directory to save checkpoints (default: `./checkpoints`).

Each checkpoint saves the model state, optimizer state, and a list of epoch losses.

### Generation

Generate new images using a trained checkpoint:

```bash
python pytorch_md4.py generate --checkpoint_path ./checkpoints/checkpoint_epoch_20.pth --num_images 16 --reverse_steps 1000 --output_path generated.png
```

#### Arguments:

- `--checkpoint_path`: Path to the checkpoint file.
- `--num_images`: Number of images to generate.
- `--reverse_steps`: Number of reverse diffusion steps.
- `--output_path`: File path to save the generated image grid.
- `--get_samples_history`: *(Optional)* If provided, the full generation history (each diffusion step) is saved as a series of images.

