"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Main script options
"""
import argparse


def get_opts():
    # Create an ArgumentParser object for cmd line argument parsing
    parser = argparse.ArgumentParser(
        description="ScriptChain Health Assessment Experiments"
    )

    # --num_texts argument: specifies the number of texts for generating a dummy dataset
    parser.add_argument(
        "--num_texts",
        type=int,
        default=1000,
        help="for generating dummy dataset"
    )

    # --pe_learnable argument: specifies whether to use learnable positional embedding or not
    parser.add_argument(
        "--pe_learnable",
        default=False,
        action="store_true",
        help="whether to use learnable positional embedding"
    )

    # --run_name argument: specifies the run name tag for wandb
    parser.add_argument(
        "--run_name",
        type=str,
        default="Run using non learnable positional embedding",
        help="run name tag for wandb",
    )

    # --batch_size argument: specifies the batch size used to run one epoch
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch Size",
    )

    # --seq_lens argument: specifies the list of sequence lengths used 
    # in the first part of the experiment
    parser.add_argument(
        "--seq_lens",
        nargs="+",
        default=[2**i for i in range(6, 10)],
        help="used in first part of experiment",
    )

    # --fixed_self_attn_layers argument: specifies the number of self-attention layers 
    # for stacking, used in the first part of the experiment
    parser.add_argument(
        "--fixed_self_attn_layers",
        type=int,
        default=8,
        help="number of self attention layers for stacking, used in first part of experiment",
    )

    # --fixed_seq_len argument: specifies the sequence length used to 
    # run the second part of the experiment
    parser.add_argument(
        "--fixed_seq_len",
        type=int,
        default=64,
        help="used to run second part of experiment",
    )

    # --self_attn_layers argument: specifies the list of numbers of self-attention 
    # layers used in the second part of the experiment
    parser.add_argument(
        "--self_attn_layers",
        nargs="+",
        default=[2**i for i in range(1, 8)],
        help="used in second part of experiment",
    )

    # Parse the command-line arguments and store the values in the `opts` variable
    opts = parser.parse_args()

    # Convert the sequence lengths and numbers of self-attention layers to integers
    opts.seq_lens = [int(item) for item in opts.seq_lens]
    opts.self_attn_layers = [int(item) for item in opts.self_attn_layers]
    
    # Return the obtained options
    return opts
