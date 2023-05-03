import torch
import torch.nn as nn

from src.models.old.navar import NAVAR


class NAVAR_MASKED(NAVAR):
    def __init__(self, num_nodes: int, kernel_size: int, n_layers: int,
                 hidden_dim: int, lambda1: float, dropout: float = 0.0):
        super(NAVAR_MASKED, self).__init__(num_nodes, kernel_size, n_layers, hidden_dim, lambda1, dropout)
        self.num_nodes = num_nodes
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels=num_nodes * num_nodes,
                                                    out_channels=num_nodes * hidden_dim,
                                                    kernel_size=kernel_size, groups=num_nodes))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(in_channels=num_nodes * hidden_dim,
                                                    out_channels=num_nodes * num_nodes,
                                                    kernel_size=kernel_size, groups=num_nodes, bias=False))
        self.receptive_field = 2 * kernel_size - 1
        pad = nn.ZeroPad2d(((kernel_size - 1), 0, 0, 0))
        self.convolutions = nn.Sequential(pad, self.conv1, nn.ReLU(), nn.Dropout(dropout), pad, self.conv2)

        self.biases = nn.Parameter(torch.empty(num_nodes, 1))
        self.attention = nn.Parameter(torch.empty(num_nodes * num_nodes, 1))

        attention_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool).fill_diagonal_(True)
        data_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
        for i in range(num_nodes):
            data_mask[i, :i + 1] = False

        self.attention_mask = attention_mask.reshape(num_nodes * num_nodes, 1)
        self.data_mask = data_mask.reshape(num_nodes * num_nodes, 1)

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        nn.init.constant_(self.biases, 0.00001)
        nn.init.constant_(self.attention, 0.001)

    def to(self, device, **kwargs):
        self.data_mask = self.data_mask.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return super().to(device=device, **kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        """
        x: Tensor of size (batch_size, num_nodes, time_steps)
        :returns:
            Tensor 'predictions' (batch_size, num_nodes, time_steps),
            Tensor 'contributions' (batch_size, num_nodes, num_nodes, time_steps),
            Tensor 'attentions' (num_nodes, num_nodes),
        """
        bs = x.shape[0]
        # attn: (num_nodes * num_nodes, 1)
        attention = self.attention * self.data_mask + self.attention_mask

        # x: (batch_size, num_nodes * num_nodes, time_steps)
        x = x.repeat(1, self.num_nodes, 1) * attention

        # x: (bs, num_nodes * num_nodes, time_steps) -> (bs, num_nodes * num_nodes, time_steps)
        x = self.convolutions(x)

        # x: (bs, num_nodes, num_nodes, time_steps)
        contributions = x.view(bs, self.num_nodes, self.num_nodes, -1)

        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases

        return predictions, contributions, attention.abs().view(self.num_nodes, self.num_nodes)


if __name__ == '__main__':
    from src.models.old.navar import test_model
    test_model(NAVAR_MASKED)