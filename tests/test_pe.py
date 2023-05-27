"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Sanity Check for Positional Encoding Layer
"""
import sys
from pathlib import Path

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

import torch
from positionalemb import PositionalEncoding, LearnablePositionalEncoding


if __name__ == "__main__":
    pos_encoding = PositionalEncoding(seq_len=10, in_dim=4)
    is_trainable = any(param.requires_grad for param in pos_encoding.parameters())
    assert not is_trainable

    x = torch.arange(80).reshape((2, 10, 4))
    out = pos_encoding(x)

    pos_encoding = LearnablePositionalEncoding(seq_len=10, in_dim=4)
    is_trainable = any(param.requires_grad for param in pos_encoding.parameters())
    
    assert is_trainable
    out = pos_encoding(x)

