#!/usr/bin/env python
"""
MD4Generation for Discrete Diffusion on MNIST

This script demonstrates:
  - Creating the MD4Generation, U-Net model, and optimizer.
  - Loading MNIST with basic data augmentation (random horizontal flipping) and binarization.
  - Training the model for N epochs with checkpointing every C epochs.
  - A generation procedure that loads a checkpoint and generates T images using R reverse steps.
  - Saving the generated images (and optionally a history of the generation process).

Usage:
  Training:
    python md4_main.py train --epochs 20 --checkpoint_interval 5 --batch_size 64 --learning_rate 1e-3

  Generation:
    python md4_main.py generate --checkpoint_path ./checkpoints/checkpoint_epoch_20.pth \
      --num_images 16 --reverse_steps 1000 --output_path generated.png
"""

import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
import torchvision.utils as vutils

import model.unet as unet

#############################################
# Helper function and MD4Generation class
#############################################

def match_last_dims(data, size):
    """
    Repeat a 1D tensor so that its last dimensions [1:] match `size[1:]`.
    Useful for working with batched data.
    """
    assert len(data.size()) == 1, "Data must be 1-dimensional (one value per batch)"
    for _ in range(len(size) - 1):
        data = data.unsqueeze(-1)
    return data.repeat(1, *(size[1:]))

class MD4Generation:
    def __init__(self, device=None, **kwargs):
        self.device = 'cpu' if device is None else device
        self.mask_number = 2

        # A masking schedule function mapping t∈[0,1] → [0,1]
        self.masking_schedule = lambda t: 1 - torch.cos((1 - t) * torch.pi / 2)
        # Cross-entropy loss weight: note the tan makes the loss positive.
        self.ce_loss_weight = lambda t: torch.pi * torch.tan((1 - t) * torch.pi / 2) / 2

    def sample(self, models, shape, get_sample_history=False, reverse_steps=1000, progress=True, **kwargs):
        
        print('Generating data...')
        
        # Initialize x_t as entirely masked.
        xt = (torch.ones(shape, device=self.device) * self.mask_number).int()
        samples = [xt.clone().detach()] if get_sample_history else []

        model = models['default']
        model.eval()
        with torch.inference_mode():
            # Discretize the time interval [0,1] into (reverse_steps + 1) points.
            t = torch.linspace(0, 1, reverse_steps + 1, device=self.device)
            progress_bar = (lambda x: x) if not progress else tqdm

            for i in progress_bar(range(reverse_steps, 1, -1)):
                # Create a time tensor for the batch.
                ti = torch.ones(shape[0], device=self.device) * t[i]
                si = torch.ones(shape[0], device=self.device) * t[i - 1]

                # First prediction pass.
                pred = model(xt, ti)
                pred = torch.sigmoid(pred)
                alphat = self.masking_schedule(ti)
                alphas = self.masking_schedule(si)
                first_factor = (alphas - alphat) / (1 - alphat)
                # second_factor is computed but not used further in this code.
                second_factor = (1 - alphas) / (1 - alphat)

                # Identify the masked pixels.
                mask_t = (xt == self.mask_number)
                mask_prob = match_last_dims(first_factor, xt.shape)
                to_unmask_xt = torch.rand(xt.shape, device=self.device) < mask_prob

                # Second prediction pass to sample new categories.
                pred = model(xt, ti)
                pred = torch.sigmoid(pred)
                predicted_cat = (torch.rand_like(pred) < pred).int()

                # Update the masked positions selected for unmasking.
                xt[mask_t & to_unmask_xt] = predicted_cat[mask_t & to_unmask_xt]

                if get_sample_history:
                    samples.append(xt.clone().detach())

        # Set any remaining masked entries, equal to mask_number, to 0.
        print('Data generation done. Setting remaining masked entries to 0...')
        xt[xt == self.mask_number] = 0

        return xt if not get_sample_history else torch.stack(samples)

    def training_losses(self, models, x_start, model_kwargs=None, **kwargs):
        model = models['default']
        x_start = x_start.to(self.device)
        x_start_clone = x_start.clone().detach()

        # Sample a time t for each example:
        t = torch.rand(1, device=x_start.device) + (1 + torch.arange(x_start.shape[0], device=x_start.device)).float() / x_start.shape[0]
        t = torch.fmod(t, 1)  # Wrap around using modulo 1.

        mask_prob = 1 - self.masking_schedule(t)
        mask_prob = match_last_dims(mask_prob, x_start.shape)
        mask = torch.rand_like(x_start) < mask_prob
        x_start[mask] = self.mask_number  # Mask out selected pixels.

        # Compute the prediction and BCE loss.
        pred = model(x_start, t)
        pred = torch.sigmoid(pred)
        weights = self.ce_loss_weight(t)
        weights = match_last_dims(weights, x_start.shape)
        loss = weights * F.binary_cross_entropy(pred, x_start_clone, reduction='none')
        loss = loss[mask]
        loss_shape = loss.shape
        loss = loss.sum(dim=list(range(1, len(loss.shape)))).mean()
        seq_len = torch.prod(torch.tensor(loss_shape, device=self.device))
        alpha_0 = self.masking_schedule(torch.tensor([0.0], device=self.device))
        recon_loss = seq_len * (1 - alpha_0) * np.log(2)  # 2 is the vocabulary size.
        loss = loss / (np.log(2) * seq_len) + recon_loss
        return {'loss': loss}

#############################################
# Placeholder functions for model/optimizer/device
#############################################


def create_unet():
    channels = 1
    out_channels = 1
    
    first_layer_embedding = True
    embedding_dim = 3 # MD4 needs a value for masks, so set of values is {0, 1, 2}
    output_dim = 1 # We only output a single probability value
    
    model = unet.UNetModel(
            in_channels=channels,
            model_channels=32,
            out_channels= out_channels,
            num_res_blocks=2,
            attention_resolutions= [2, 4],# tuple([2, 4]), # adds attention at image_size / 2 and /4
            dropout= 0.0,
            channel_mult= [1, 2, 2, 2], # divides image_size by two at each new item, except first one. [i] * model_channels
            dims = 2, # for images
            num_classes= None,#
            num_heads=4,
            num_heads_upsample=-1, # same as num_heads
            use_scale_shift_norm=True,
            first_layer_embedding=first_layer_embedding,
            embedding_dim= embedding_dim,
            output_dim = output_dim,
        )
    return model

def create_optimizer(model, lr=1e-3):
    """
    Returns an Adam optimizer for the model.
    """
    return optim.AdamW(model.parameters(), 
                                lr=lr, 
                                betas=(0.9, 0.999))
    

def get_device():
    """
    Returns the available device ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


#############################################
# Data loader for MNIST with augmentation
#############################################

def create_dataloader(batch_size):
    """
    Returns a DataLoader for MNIST with basic augmentation:
      - Random horizontal flip.
      - Binarization (threshold at 0.5).
    """
    transform = transforms.Compose([
        transforms.Resize(32), # scale images from 28x28 to 32x32 to fit the unet
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Binarize the image: pixels > 0.5 become 1, else 0.
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

#############################################
# Training loop with checkpointing
#############################################

def train(num_epochs, checkpoint_interval, batch_size, learning_rate, checkpoint_dir):
    device = get_device()
    print("Using device:", device)

    # Create objects.
    md4_gen = MD4Generation(device=device)
    model = create_unet().to(device)
    optimizer = create_optimizer(model, lr=learning_rate)
    dataloader = create_dataloader(batch_size)

    model.train()
    epoch_losses = []
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for batch_idx, (data, _) in tqdm(enumerate(dataloader)):
            data = data.to(device) 
            optimizer.zero_grad()

            # Compute the training loss.
            losses_dict = md4_gen.training_losses(models={'default': model}, x_start=data)
            loss = losses_dict['loss']
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if batch_idx % 100 == 0:
            #     print(f"Epoch [{epoch}] Batch [{batch_idx}] Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch}] Average Loss: {avg_loss:.4f}")

        # Save a checkpoint every checkpoint_interval epochs.
        if epoch % checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_losses': epoch_losses,
            }, checkpoint_path)
            print("Saved checkpoint to", checkpoint_path)

    print("Training finished.")

#############################################
# Generation procedure and image saving
#############################################

def generate(checkpoint_path, num_images, reverse_steps, get_samples_history, output_path):
    device = get_device()
    print("Using device:", device)

    # Create MD4Generation and the model.
    md4_gen = MD4Generation(device=device)
    model = create_unet().to(device)

    # Load the checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.inference_mode():
        samples = md4_gen.sample(
            models={'default': model},
            shape=(num_images, 1, 32, 32),
            reverse_steps=reverse_steps,
            get_sample_history=get_samples_history,
            progress=True
        )

    # Save the generated images.
    save_images(samples, output_path, get_samples_history)
    print("Saved generated images to", output_path)

def save_images(samples, output_path, get_samples_history=False):
    """
    Save a grid of images to output_path. If get_samples_history is True,
    also save the full history as separate images in a folder.
    """
    
    # copy samples and convert to float
    float_samples = samples.clone().detach().float()
    
    if get_samples_history:
        # Save final samples.
        final_samples = float_samples[-1]
        vutils.save_image(final_samples, output_path, nrow=int(math.sqrt(final_samples.size(0))), normalize=True)
        # Also save each generation step as an image.
        history_dir = os.path.splitext(output_path)[0] + "_history"
        os.makedirs(history_dir, exist_ok=True)
        for i, step_samples in enumerate(float_samples):
            step_path = os.path.join(history_dir, f"step_{i:04d}.png")
            vutils.save_image(step_samples, step_path, nrow=int(math.sqrt(step_samples.size(0))), normalize=True)
    else:
        vutils.save_image(float_samples, output_path, nrow=int(math.sqrt(float_samples.size(0))), normalize=True)

#############################################
# Main entry point with argument parsing
#############################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MD4Generation Training and Generation Script")
    subparsers = parser.add_subparsers(dest="command", help="Commands: train or generate")

    # Sub-parser for training.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (N)")
    train_parser.add_argument("--checkpoint_interval", type=int, default=5, help="Checkpoint interval (every C epochs)")
    train_parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")

    # Sub-parser for generation.
    gen_parser = subparsers.add_parser("generate", help="Generate images using a checkpoint")
    gen_parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    gen_parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate (T)")
    gen_parser.add_argument("--reverse_steps", type=int, default=1000, help="Number of reverse diffusion steps (R)")
    gen_parser.add_argument("--get_samples_history", action="store_true", help="Save the full generation history")
    gen_parser.add_argument("--output_path", type=str, default="generated.png", help="Path to save the generated image grid")

    args = parser.parse_args()

    if args.command == "train":
        train(
            num_epochs=args.epochs,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            checkpoint_dir=args.checkpoint_dir
        )
    elif args.command == "generate":
        generate(
            checkpoint_path=args.checkpoint_path,
            num_images=args.num_images,
            reverse_steps=args.reverse_steps,
            get_samples_history=args.get_samples_history,
            output_path=args.output_path
        )
    else:
        parser.print_help()