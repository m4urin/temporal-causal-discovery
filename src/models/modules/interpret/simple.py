import torch
import torch.nn as nn

from src.models.modules.interpret.interpret_module import InterpretModule
from src.models.modules.temporal.temporal_variational_layer import TemporalVariationalLayer


class SimpleDefault(InterpretModule):
    def __init__(self,
                 in_channels: int,
                 groups: int,
                 num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables)
        self.predict = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels, self.n, kernel_size=1, groups=1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of size (batch_size, in_channels, sequence_length)
        """
        return {'x': self.predict(x)}


class SimpleVar(InterpretModule):
    def __init__(self,
                 in_channels: int,
                 groups: int,
                 num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables, use_variational_layer=True)
        self.predict = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=1),
            nn.ReLU(),
            TemporalVariationalLayer(in_channels, self.n, groups=1)
        )

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of size (batch_size, in_channels, sequence_length)
        """
        x, mu, std, kl_loss = self.predict(x)
        return {
            'x': x,
            'mu': mu,
            'std': std,
            'loss': kl_loss
        }


class SimpleInstant(InterpretModule):
    def __init__(self,
                 in_channels: int,
                 groups: int,
                 num_external_variables: int):
        super().__init__(in_channels, groups, num_external_variables)
        self.predict_t0 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels, self.n, kernel_size=1, groups=1)
        )
        self.predict_t1 = nn.Sequential(
            nn.Conv1d(self.n * self.dim, self.n * self.dim, kernel_size=1, groups=1),
            nn.ReLU(),
            nn.Conv1d(self.n * self.dim, self.n, kernel_size=1, groups=1)
        )

        mask = 1 - torch.eye(self.n, self.n, dtype=torch.int).reshape(1, self.n, self.n, 1)
        self.register_buffer('mask', mask)

        self.pad = nn.ConstantPad1d((0, 1), 0)

    def calc(self, x: torch.Tensor, layer, biases=None, mask=None):
        batch_size, _, seq_len = x.size()

        if mask is not None:
            x.reshape()
        else:
            pass

        x = layer(x)

        if mask is None:
            x = x.reshape(batch_size, self.groups, self.n, seq_len)
        else:
            x = x.reshape(batch_size, self.n, self.n, seq_len).masked_fill(self.mask == 0, 0)

        attn = x[:, :self.n]
        attn_ext = x[:, self.n:]

        x = x.sum(dim=1)  # (batch_size, n, seq_len)

        if biases is not None:
            x += biases

        return x, attn, attn_ext

    def forward(self, x: torch.Tensor):
        """
        x: Tensor of size (batch_size, in_channels, sequence_length)
        """
        batch_size, _, seq_len = x.size()
        x = self.pad(x)

        x0, attn, attn_ext = self.calc(x[..., :-1], self.contributions_t0, biases=self.biases)

        x = x.reshape(batch_size, self.group, self.dim, seq_len + 1)[:, :self.n, :, 1:].reshape(batch_size, -1, seq_len)
        x1, attn1, _ = self.calc(x, self.contributions_t1, mask=self.mask)

        return {
            'x': x0 + x1,
            'attn': attn,
            'attn_instantaneous': attn1,
            'attn_external_variables': attn_ext
        }

if __name__ == '__main__':
    a = nn.Conv1d(6, 5*3, groups=3, kernel_size=1, bias=False)
    b = nn.Conv1d(6, 5, groups=1, kernel_size=1, bias=False)

    for n, p in a.named_parameters():
        print(n, p.size())
    for n, p in b.named_parameters():
        print(n, p.size())

    a.weight.data = b.weight.data.reshape(5, 3, 2, 1).permute(1, 0, 2, 3).reshape(15, 2, 1)

    data = torch.rand(1, 6, 1)

    a = a(data)
    b = b(data)
    print(a.size())
    print(b.size())

    print(a.reshape(1, 3, 5, 1).sum(dim=1))
    print(b)
