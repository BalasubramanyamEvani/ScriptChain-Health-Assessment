"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Utility Functions
"""
import torch
from tqdm import tqdm
import string
import time
import gc
from collections import defaultdict


def dummy_dictionary():
    """
    Create a dummy dictionary for experiments
    This maps each letter in the alphabet to its corresponding index.

    Returns:
    - dict: A dictionary mapping each letter to its index.
    """
    dictionary = {}
    for i, letter in enumerate(string.ascii_letters):
        dictionary[letter] = i
    return dictionary


def get_device():
    """
    Get the device available for computation.

    Returns:
    - str: The name of the device ("cuda" if CUDA is available, otherwise "cpu").
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def run(model, criterion, optimizer, dataloader, device, epochs, log_grad=False):
    """
    Train model using the specified dataloader, optimizer, and criterion.

    Parameters:
    - model (nn.Module): The model to train.
    - dataloader (DataLoader): Dummy dataloader.
    - optimizer (Optimizer): The optimizer to use for training.
    - criterion (Loss): The loss criterion to optimize.
    - device (str or torch.device): The device to perform training on ("cuda" or "cpu").
    - epochs (int): The number of training epochs. For experiments 1 is used.
    - log_grad (bool, optional): Whether to log the mean gradients of the model parameters. Defaults to False.
        This is used to understand gradient flow in lower layers in experiment 2

    Returns:
    - Tuple[int, dict]: A tuple containing the execution time in seconds and a dictionary of mean gradients
      of the model parameters (if log_grad is True).
    """
    start_time = time.time()
    grad_accumulator = defaultdict(int)
    tot = 0

    model.train()
    for _ in range(epochs):
        for _, (x, y) in enumerate(tqdm(dataloader)):
            tot += 1

            x = x.to(device)
            y = y.to(device).reshape(-1, 1)

            optimizer.zero_grad()

            out = model(x)
            loss = criterion(out, y)
            loss.backward()

            if log_grad:
                for name, param in model.named_parameters():
                    if (param.requires_grad) and ("bias" not in name):
                        grad_accumulator[name] += param.grad.abs().mean()
            del x
            del y
            del out

            optimizer.step()

    gc.collect()
    torch.cuda.empty_cache()

    for key in grad_accumulator.keys():
        grad_accumulator[key] = grad_accumulator[key] / tot

    end_time = time.time()
    execution_time = round(end_time - start_time)

    return execution_time, grad_accumulator
