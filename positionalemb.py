"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Positional Encodings - Default and Learnable
"""
import torch


class PositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, in_dim, dropout_p=0.1) -> None:
        """
        Non Learnable positional encoding layer.

        Parameters:
        - seq_len (int): The length of the input sequence.
        - in_dim (int): The input dimension.
        - dropout_p (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        pe = torch.zeros((1, seq_len, in_dim))
        nume = torch.arange(seq_len).float().reshape(-1, 1)
        denom = torch.exp(
            torch.arange(0, in_dim, 2).float()
            * (-torch.log(torch.tensor(10000).float()) / in_dim)
        )
        res = nume * denom
        pe[:, :, 0::2] = torch.sin(res)
        pe[:, :, 1::2] = torch.cos(res)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Forward pass of the non learnable positional encoding layer.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after adding positional encoding.
        """
        x = x + self.pe.to(x.device)
        out = self.dropout(x)
        return out


class LearnablePositionalEncoding(torch.nn.Module):
    def __init__(self, seq_len, in_dim, dropout_p=0.1) -> None:
        """
        Learnable positional encoding layer.

        Parameters:
        - seq_len (int): The length of the input sequence.
        - in_dim (int): The input dimension.
        - dropout_p (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.positions = torch.arange(seq_len).unsqueeze(0)
        self.pe = torch.nn.Embedding(seq_len, in_dim)

    def forward(self, x):
        """
        Forward pass of the learnable positional encoding layer.

        Parameters:
        - x (torch.Tensor): The input tensor.

        Returns:
        - torch.Tensor: The output tensor after adding positional encoding.
        """
        x = x + self.pe(self.positions.to(x.device))
        out = self.dropout(x)
        return out
