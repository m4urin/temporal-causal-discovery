import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, n_variables, hidden_dim, n_ensembles=1, max_length=2000):
        super().__init__()
        self.n_ensembles = n_ensembles
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv1d(
            in_channels=n_variables,
            out_channels=n_variables * n_ensembles * hidden_dim,
            kernel_size=1,
            groups=n_variables
        )

        # Create the positional embedding
        pos_embedding = torch.zeros(1, hidden_dim, max_length)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pos_embedding[0, 0::2, :] = torch.sin(position * div_term).t()
        pos_embedding[0, 1::2, :] = torch.cos(position * div_term).t()
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        # Add the positional embedding to x
        batch_size, _, seq_len = x.size()
        x = self.conv(x)  # (batch_size, n_var * n_ensembles * hidden_dim, seq_len)
        x = x.reshape(-1, self.hidden_dim, seq_len) + self.pos_embedding[..., :seq_len]
        return x.reshape(batch_size, -1, seq_len)  # (batch_size, n_var * n_ensembles * hidden_dim, seq_len)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.plot(PositionalEmbedding(1, 10, max_length=200).pos_embedding[0].t())
    plt.show()
