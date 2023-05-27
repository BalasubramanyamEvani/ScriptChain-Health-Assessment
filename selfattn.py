"""
Author: Balasubramanyam Evani
Date: 2023-05-27

Description: Multi-Head Self Attention Layer
"""
import torch


class MultiHeadSelfAttentionLayer(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=8, dropout_p=0.1) -> None:
        """
        Multi-head self-attention layer.

        Parameters:
        - in_dim (int): The input dimension.
        - hidden_dim (int): The hidden dimension for each head.
        - num_heads (int, optional): The number of attention heads. Defaults to 8.
        - dropout_p (float, optional): The dropout probability. Defaults to 0.1.
        """
        super(MultiHeadSelfAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.out_dim = self.in_dim
        self.dk = torch.tensor(self.hidden_dim).float()

        self.vw = torch.nn.Linear(self.in_dim, self.hidden_dim * self.num_heads)
        self.qw = torch.nn.Linear(self.in_dim, self.hidden_dim * self.num_heads)
        self.kw = torch.nn.Linear(self.in_dim, self.hidden_dim * self.num_heads)

        self.softmax_layer = torch.nn.Softmax(dim=-1)
        self.droput = torch.nn.Dropout(dropout_p)
        self.fc_out = torch.nn.Linear(self.hidden_dim * num_heads, self.out_dim)

    def forward(self, x):
        """
        Forward pass of the multi-head self-attention layer.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (batch_size, seq_len, emb_size).

        Returns:
        - torch.Tensor: The output tensor of shape (batch_size, seq_len, emb_size).
        """
        self.dk = self.dk.to(x.device)
        batch_size, seq_len, emb_size = x.shape

        queries_proj = self.qw(x)
        keys_proj = self.kw(x)
        values_proj = self.vw(x)

        queries_proj = queries_proj.reshape(
            batch_size, seq_len, self.num_heads, self.hidden_dim
        ).transpose(1, 2)

        keys_proj = keys_proj.reshape(
            batch_size, seq_len, self.num_heads, self.hidden_dim
        ).transpose(1, 2)

        values_proj = values_proj.reshape(
            batch_size, seq_len, self.num_heads, self.hidden_dim
        ).transpose(1, 2)

        attn_scores = torch.matmul(queries_proj, keys_proj.transpose(3, 2))
        attn_scores = attn_scores / torch.sqrt(self.dk)

        attn_softmax_scores = self.softmax_layer(attn_scores)
        attn_out = torch.matmul(attn_softmax_scores, values_proj)

        attn_concat = attn_out.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.hidden_dim
        )

        linear_out = self.fc_out(attn_concat)
        out = self.droput(linear_out)
        return out
