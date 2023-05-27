"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Main Script
Does two experiments:

Experiment for Hypothesis 1: 
As we increase the number of multi-head self-attention layers, the overall computing 
should be computationally expensive. Since, the self-attention operation has a quadratic 
time complexity concerning the sequence length. I expect to observe prohibitively slower 
performance or requirement of significant computational resources for longer sequences.

Experiment for Hypothesis 2: Only stacking Multi-head attention layers might result in a 
vanishing gradient problem. The original transformer paper used MHA in conjecture with 
Residual connections. Hence, I hope to see average gradient value per layer become smaller 
and smaller as gradients flow toward the initial layers.
"""
import torch
import wandb

from utils import dummy_dictionary, get_device, run
from dataset import DummyDataset
from opts import get_opts
from encoder import DummyClassifier


if __name__ == "__main__":
    wandb.login(key="")
    opts = get_opts()

    print(opts)

    # Initialize Weights & Biases (wandb) with the specified project name
    wandb.init(project="script-chain-health-exps")

    # Update the wandb configuration with the options obtained
    wandb.config.update(opts)

    # Set the run name for wandb to the specified run_name from the options
    wandb.run.name = opts.run_name

    pe_learnable = opts.pe_learnable
    seq_lens = opts.seq_lens
    batch_size = opts.batch_size
    num_self_attn_layers = opts.fixed_self_attn_layers
    num_texts = opts.num_texts

    # Get the device to be used for training
    device = get_device()

    # Create a dummy dictionary
    dictionary = dummy_dictionary()

    # Obtain the vocabulary size from the dummy dictionary
    vocab_size = len(dictionary)

    # Exp 1: Different Seq Lengths
    # Iterate over each sequence length in seq_lens
    for seq_len in seq_lens:
        # Create a dummy dataset with the specified number of texts and sequence length
        dataset = DummyDataset(num_texts=num_texts, seq_len=seq_len, dictionary=dictionary)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
        )
        # Create a dummy classifier model with the specified number of self-attention layers, 
        # sequence length, vocab size, and pe_learnable flag
        model = DummyClassifier(
            enc_self_attn_layers=num_self_attn_layers,
            seq_len=seq_len,
            vocab_size=vocab_size,
            pe_learnable=pe_learnable,
        )
        model = model.to(device)
        criterion = torch.nn.BCELoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-3,
            momentum=0.9,
            weight_decay=0,
        )
        # Run the model for one epoch and collect the execution time
        execution_time, _ = run(model, criterion, optimizer, dataloader, device, epochs=1)
        
        # Log the experiment results to wandb for Exp 1
        wandb.log({"Exp1: Input Sequence length": seq_len, "Exp1: Execution time (s)": execution_time})

    fixed_seq_len = opts.fixed_seq_len
    self_attn_layers = opts.self_attn_layers

    # Exp 2: Different Self Attention layers
    # Create a dataset with the fixed sequence length
    dataset = DummyDataset(num_texts=num_texts, seq_len=fixed_seq_len, dictionary=dictionary)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

    # Iterate over each number of self-attention layers in self_attn_layers
    for num_layers in self_attn_layers:
        # Create a dummy classifier model with the specified number of self-attention layers, 
        # fixed sequence length, vocab size, and pe_learnable flag
        model = DummyClassifier(
            enc_self_attn_layers=num_layers,
            seq_len=fixed_seq_len,
            vocab_size=vocab_size,
            pe_learnable=pe_learnable,
        )
        model = model.to(device)
        criterion = torch.nn.BCELoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-3,
            momentum=0.9,
            weight_decay=0,
        )
        # Run the model training for one epoch, 
        # collect the execution time, and retrieve the gradient accumulator
        execution_time, grad_accumulator = run(model, criterion, optimizer, dataloader, device, epochs=1, log_grad=True)
        
        log_info = {
            "Exp2: Num Stacked Self. Attn. Layers": num_layers,
            "Exp2: Execution time (s)": execution_time
        }

        # Log the gradients for specific keys (lower layer and learnable embedding)
        for key in grad_accumulator.keys():
            if "embedding" in key or "enc_modules.0" in key or "encoder.pe.pe.weight" in key:
                log_info[f"Exp2: [Gradient] {key}"] = grad_accumulator[key]

        # Log the experiment results to wandb for Exp 2
        wandb.log(log_info)

    wandb.run.finish()
