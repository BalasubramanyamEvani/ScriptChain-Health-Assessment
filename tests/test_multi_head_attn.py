"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Sanity Check for Multi-Head Self Attention Layer
"""
import sys
from pathlib import Path

sys.path.append(str(Path(sys.argv[0]).resolve().parent.parent))

from selfattn import MultiHeadSelfAttentionLayer
from torchinfo import summary


if __name__ == "__main__":
    multhead_attn = MultiHeadSelfAttentionLayer(in_dim=512, hidden_dim=512)
    summary(multhead_attn, input_size=(8, 100, 512))
