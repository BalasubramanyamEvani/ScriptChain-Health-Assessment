"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Main Script
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

    wandb.init(project="script-chain-health-exps")
    wandb.config.update(opts)
    wandb.run.name = opts.run_name

    pe_learnable = opts.pe_learnable
    seq_lens = opts.seq_lens
    batch_size = opts.batch_size
    num_self_attn_layers = opts.fixed_self_attn_layers
    num_texts = opts.num_texts

    device = get_device()
    dictionary = dummy_dictionary()
    vocab_size = len(dictionary)

    # Exp 1: Different Seq Lengths

    for seq_len in seq_lens:
        dataset = DummyDataset(num_texts=num_texts, seq_len=seq_len, dictionary=dictionary)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=True,
        )
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
        execution_time, _ = run(model, criterion, optimizer, dataloader, device, epochs=1)
        wandb.log({"Exp1: Input Sequence length": seq_len, "Exp1: Execution time (s)": execution_time})

    fixed_seq_len = opts.fixed_seq_len
    self_attn_layers = opts.self_attn_layers

    # Exp 2: Different Self Attention layers
    dataset = DummyDataset(num_texts=num_texts, seq_len=fixed_seq_len, dictionary=dictionary)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )
    for num_layers in self_attn_layers:
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
        execution_time, grad_accumulator = run(model, criterion, optimizer, dataloader, device, epochs=1, log_grad=True)
        
        log_info = {
            "Exp2: Num Stacked Self. Attn. Layers": num_layers,
            "Exp2: Execution time (s)": execution_time
        }

        for key in grad_accumulator.keys():
            if "embedding" in key or "enc_modules.0" in key:
                log_info[f"Exp2: [Gradient] {key}"] = grad_accumulator[key]

        wandb.log(log_info)

    wandb.run.finish()
