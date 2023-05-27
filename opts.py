"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Main scripts options
"""
import argparse


def get_opts():
    parser = argparse.ArgumentParser(
        description="ScriptChain Health Assessment Experiments"
    )
    parser.add_argument(
        "--num_texts",
        type=int,
        default=1000,
        help="for generating dummy dataset"
    )
    parser.add_argument(
        "--pe_learnable",
        default=False,
        action="store_true",
        help="whether to use learnable positional embedding"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="Run using non learnable positional embedding",
        help="run name tag for wandb",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch Size",
    )
    parser.add_argument(
        "--seq_lens",
        nargs="+",
        default=[2**i for i in range(6, 10)],
        help="used in first part of experiment",
    )
    parser.add_argument(
        "--fixed_self_attn_layers",
        type=int,
        default=8,
        help="number of self attention layers for stacking, used in first part of experiment",
    )
    parser.add_argument(
        "--fixed_seq_len",
        type=int,
        default=64,
        help="used to run second part of experiment",
    )
    parser.add_argument(
        "--self_attn_layers",
        nargs="+",
        default=[2**i for i in range(1, 8)],
        help="used in second part of experiment",
    )
    opts = parser.parse_args()

    opts.seq_lens = [int(item) for item in opts.seq_lens]
    opts.self_attn_layers = [int(item) for item in opts.self_attn_layers]
    
    return opts
