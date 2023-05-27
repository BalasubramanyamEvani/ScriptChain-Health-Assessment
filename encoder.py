"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Stacked Encoder built using Multi-Head Attention Layers
"""
import torch
from selfattn import MultiHeadSelfAttentionLayer
from positionalemb import PositionalEncoding, LearnablePositionalEncoding


class StackedMultiHeadAttentionEncoder(torch.nn.Module):
    def __init__(self, num_layers, seq_len, pe_learnable=False, in_dim=512) -> None:
        """
        Stacked multi-head attention encoder.

        Parameters:
        - num_layers (int): The number of attention layers to stack.
        - seq_len (int): The length of the input sequence.
        - pe_learnable (bool, optional): Whether to use learnable positional encoding. Defaults to False.
        - in_dim (int, optional): The input dimension. Defaults to 512.
        """
        super(StackedMultiHeadAttentionEncoder, self).__init__()
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.pe_learnable = pe_learnable
        self.pe = None
        if not self.pe_learnable:
            self.pe = PositionalEncoding(seq_len=seq_len, in_dim=in_dim)
        else:
            self.pe = LearnablePositionalEncoding(seq_len=seq_len, in_dim=in_dim)

        self.enc_modules = []
        for _ in range(num_layers):
            self.enc_modules.append(
                MultiHeadSelfAttentionLayer(in_dim=in_dim, hidden_dim=in_dim)
            )

        self.enc_modules = torch.nn.ModuleList(self.enc_modules)

    def forward(self, x):
        """
        Forward pass of the stacked multi-head attention encoder.
        Specified Positional encoding is added before passing to the
        encoder layers

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after encoder layers.
        """
        enc_out = x + self.pe(x)
        for enc_module in self.enc_modules:
            enc_out = enc_module(enc_out)
        return enc_out


class DummyClassifier(torch.nn.Module):
    def __init__(
        self,
        enc_self_attn_layers,
        seq_len,
        vocab_size,
        pe_learnable=False,
        embedding_dim=512,
    ) -> None:
        """
        Dummy classifier model.

        Parameters:
        - enc_self_attn_layers (int): The number of encoder self-attention layers to stack.
        - seq_len (int): The length of the input sequence.
        - vocab_size (int): The size of the vocabulary.
        - pe_learnable (bool, optional): Whether to use learnable positional encoding. Defaults to False.
        - embedding_dim (int, optional): The dimension of the input embedding. Defaults to 512.
        """
        super(DummyClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.encoder = StackedMultiHeadAttentionEncoder(
            num_layers=enc_self_attn_layers,
            seq_len=seq_len,
            pe_learnable=pe_learnable,
            in_dim=embedding_dim,
        )
        self.fc_out = torch.nn.Linear(embedding_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the dummy classifier model.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after sigmoid.
        """
        x = self.embedding(x)
        enc_out = self.encoder(x)[:, -1, :]
        out = self.fc_out(enc_out)
        out = self.sigmoid(out)
        return out
