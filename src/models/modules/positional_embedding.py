import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Positional embeddings for causal variables.

    This class provides an implementation of positional embeddings for causal variables. The positional embeddings
    capture the relative positions of variables in a sentence, allowing the model to better capture the
    history-dependent noise.

    Args:
        num_variables (int): The number of variables to be embedded. Each variable will have its own embedding.
        embedding_dim (int): The size of the embedding dimension. The larger the dimension, the more information
            can be encoded in the embeddings.
        sequence_length (int): The length of the sequence to be embedded. The longer the sequence, the more positions
            need to be encoded in the positional embeddings. Note that the maximum sequence length is limited by the
            computational resources available, as the size of the positional embeddings grows linearly with the sequence
            length.
    """
    def __init__(self, num_variables: int, embedding_dim: int, sequence_length: int, batch_size: int = 1) -> None:
        super(PositionalEmbedding, self).__init__()
        self.num_variables = num_variables
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # Compute the positional encodings once in log space
        pe = torch.zeros(sequence_length, embedding_dim)
        position = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2).unsqueeze(0)
        self.register_buffer('pe', pe)

        # One embedding for each variable
        #self.embeddings = nn.Parameter(0.5 * torch.randn(batch_size, num_variables, embedding_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the current embedding with the positional encoding.

        Args:
            x (torch.Tensor): Tensor of size (batch_size, num_variables, embedding_dim)
        Returns:
            torch.Tensor: The resulting tensor of size (batch_size, num_variables * embedding_dim, sequence_length).
        """
        return (x.unsqueeze(-1) + self.pe).reshape(self.batch_size, -1, self.sequence_length)

    def __str__(self):
        return f"PositionalEmbedding(batch_size={self.batch_size}, " \
               f"num_variables={self.num_variables}, " \
               f"embedding_dim={self.embedding_dim}, " \
               f"sequence_length={self.sequence_length})"

    def __repr__(self):
        return str(self)
