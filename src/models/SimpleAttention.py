import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import trange

from src.eval.soft_roc_auc import roc_auc_score
from src.utils import count_parameters


def f3(x0, x1, x2):
    return 0.8 * x0 + 0.8 * x1 - 0.8 * x2


def generate_random_data():
    batch_size, sequence_length = 1, 300
    data = (0.5 * torch.randn(batch_size, 4, sequence_length + 1))
    # data[:, 0, [100, 400, 200, 300]] = torch.tensor([-4.5, -4, 4, 4.5])
    data[:, -1] *= 0.0

    true_function = torch.zeros_like(data)

    # compute var 1
    for i in range(1, sequence_length + 1):
        true_function[:, 1, i] = np.sin(8 * data[:, 0, i - 1])
        data[:, 1, i] += true_function[:, 1, i]

    # compute var 2
    for i in range(1, sequence_length + 1):
        true_function[:, 2, i] = np.tanh(-4.0 * data[:, 1, i - 1])
        data[:, 2, i] += true_function[:, 2, i]

    # compute var 3
    for i in range(1, sequence_length + 1):
        true_function[:, 3, i] = np.sin(f3(*[data[:, g, i - 2] for g in range(3)]))
        data[:, 3, i] += true_function[:, 3, i]

    # normalize
    d_mean, d_std = data.mean(dim=-1, keepdim=True), data.std(dim=-1, keepdim=True)
    data = (data - d_mean) / d_std
    true_function = (true_function - d_mean) / d_std

    x = data[..., :-1]
    y = data[..., 1:]
    true_function = true_function[..., 1:]

    true_attn = torch.zeros(4, 4)
    true_attn[1, 0] = 1
    true_attn[2, 1] = 1
    true_attn[3, :3] = 1

    if torch.cuda.is_available():
        x, y, true_function = x.cuda(), y.cuda(), true_function.cuda()

    return x, y, true_function, true_attn


class SimpleAttention(nn.Module):
    def __init__(self, k, n, n_layers=3, hidden_dim=32, dropout=0.2, **kwargs):
        super().__init__()
        assert n_layers > 0
        self.k = k
        self.n = n
        self.receptive_field = 2 ** n_layers

        self.hidden_dim = hidden_dim

        reg_mask = (torch.rand(1, k, n, n, 1) < 0.4).float()  #(1 / (n - 1))
        self.register_buffer("reg_mask", reg_mask)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv = []
        self.tcn = []

        for i in range(n_layers):
            conv = nn.Conv1d(
                in_channels=(k + 1) * n if i == 0 else (k + 1) * n * hidden_dim,
                out_channels=(k + 1) * n * hidden_dim,
                kernel_size=2,
                dilation=2 ** i,
                groups=(k + 1) * n
            )
            self.conv.append(conv)
            self.tcn.extend([conv, self.relu, self.dropout])

        self.tcn = nn.Sequential(*self.tcn)

        self.projection = nn.Conv1d(
            in_channels=k * n * hidden_dim,
            out_channels=k * n * (n + hidden_dim),  # n attentions and the vector h to broadcast
            kernel_size=1,
            groups=k * n
        )

        self.prediction = nn.Sequential(
            nn.Conv1d(
                in_channels=k * n * hidden_dim,
                out_channels=k * n * hidden_dim // 2,
                kernel_size=1,
                groups=k * n
            ),
            self.relu,
            nn.Conv1d(
                in_channels=k * n * hidden_dim // 2,
                out_channels=k * n,
                kernel_size=1,
                groups=k * n
            )
        )

        self.contributions = nn.Conv1d(
            in_channels=n * hidden_dim,
            out_channels=n * n,
            kernel_size=1,
            groups=n
        )
        self.biases = nn.Parameter(torch.ones(1, n, 1) * 0.001)

        self.n_params = count_parameters(self)

    def forward(self, x, temperature=1.0):

        """ TCN """
        # x: (batch_size, n, sequence_length)
        batch_size = x.size(0)

        # x: (batch_size, (k+1) * n, sequence_length)
        x = x.repeat(1, self.k + 1, 1)

        # x: (batch_size, (k+1) * n * hidden_dim, sequence_length)
        x = self.tcn(x)

        nd = self.n * self.hidden_dim

        """ NAVAR """
        # contr: (batch_size, n, n, sequence_length)
        contributions = self.contributions(x[:, :nd, :]).reshape(batch_size, self.n, self.n, -1)
        # pred: (batch_size, n, sequence_length)
        navar_predictions = contributions.sum(dim=1) + self.biases  # (bs, n, seq)

        """ Attentions """
        # x: (batch_size, k, n, (hidden_dim + n), sequence_length)
        x = self.projection(x[:, nd:, :]).reshape(batch_size, self.k, self.n, self.n + self.hidden_dim, -1)

        # attentions: (batch_size, k, n, n, sequence_length)
        attentions = torch.softmax(x[..., :self.n, :] / temperature, dim=-2)

        # context: (batch_size, k, n, hidden_dim, sequence_length)
        context = x[..., self.n:, :]

        # x: (batch_size, k, n, hidden_dim, sequence_length)
        x = torch.einsum('bkijt, bkjdt -> bkidt', attentions, context)

        # x: (batch_size, k * n * hidden_dim, sequence_length)
        x = x.reshape(batch_size, self.k * self.n * self.hidden_dim, -1)

        # x: (batch_size, k, n, sequence_length)
        attn_predictions = self.prediction(x).reshape(batch_size, self.k, self.n, -1)

        return {
            'navar_predictions': navar_predictions,
            'attn_predictions': attn_predictions,
            'contributions': contributions,
            'attentions': attentions
        }

    def loss_function(self, y_true, navar_predictions, attn_predictions, contributions, attentions):
        # kernel_weights = torch.stack([conv.weight.reshape(self.groups, -1, 2).abs().mean(dim=1) for conv in self.conv])
        # weight_loss = (kernel_weights ** 2).mean()

        error_navar = ((navar_predictions - y_true) ** 2).mean()
        reg_navar = contributions.abs().mean()

        error_attn = ((attn_predictions - y_true.unsqueeze(1)) ** 2).mean()
        attn_reg = (attentions * self.reg_mask).mean()

        return error_navar + 0.5 * reg_navar + error_attn + 0.02 * attn_reg


def train_model(model, x, y, epochs=3000, lr=1e-3, weight_decay=1e-5, disable_tqdm=False):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    t_true = y[..., model.receptive_field - 1:]

    pbar = trange(epochs, disable=disable_tqdm)
    for epoch in pbar:
        optimizer.zero_grad()
        model_output = model(x)
        loss = model.loss_function(t_true, **model_output)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            pbar.set_description(f"Loss: {loss.item():.4f}")
        if epoch == int(0.8 * epochs):
            optimizer = optim.AdamW(model.parameters(), lr=lr / 10, weight_decay=weight_decay)

        pbar.update()

    with torch.no_grad():
        model.eval()
        model_output = model(x, temperature=0.9)  # more sharp predictions

    model_output['attn_predictions'] = model_output['attn_predictions'].mean(dim=1)  # over k
    model_output['contributions'] = model_output['contributions'].std(dim=-1).transpose(-1, -2)  # over sequence length
    model_output['attentions_std'] = model_output['attentions'].std(dim=(1, -1))  # over k and sequence length
    model_output['attentions'] = model_output['attentions'].mean(dim=(1, -1))  # over k and sequence length
    return {k: v[0].cpu() for k, v in model_output.items()}


def print_matrices_scores(true_attentions, contributions, attentions, attentions_std, **kwargs):
    print('\n----- NAVAR -----\n')
    print(f'Contributions: {roc_auc_score(true_attentions, contributions)[2].item():.2f}')
    print(contributions)

    print('\n----- TAM -----\n')
    print(f'Mean: {roc_auc_score(true_attentions, attentions)[2].item():.2f}')
    print(attentions)
    print('Std:')
    print(attentions_std)

    max_ = attentions.max(dim=-1, keepdim=True).values
    attn = attentions / max_
    attn_std = attentions_std / max_

    final1 = (attn - attn_std).clamp(min=0)
    print(f'mean - 1x std: {roc_auc_score(true_attentions, final1)[2].item():.2f}')
    print(final1)

    final2 = (attn - 2 * attn_std).clamp(min=0)
    print(f'mean - 2x std: {roc_auc_score(true_attentions, final2)[2].item():.2f}')
    print(final2)


def print_matrices(contributions, attentions, attentions_std, **kwargs):
    print('\n----- NAVAR -----\n')
    print(f'Contributions')
    print(contributions)

    print('\n----- TAM -----\n')
    print(f'Mean:')
    print(attentions)
    print('Std:')
    print(attentions_std)

    max_ = attentions.max(dim=-1, keepdim=True).values
    attn = attentions / max_
    attn_std = attentions_std / max_

    final1 = (attn - attn_std).clamp(min=0)
    print(f'mean - 1x std:')
    print(final1)

    final2 = (attn - 2 * attn_std).clamp(min=0)
    print(f'mean - 2x std:')
    print(final2)


def test():
    x, y, true_functions, true_attn = generate_random_data()
    # plt.scatter(f3(x[0, 0, :-1], x[0, 1, :-1], x[0, 2, :-1]).cpu(), y[0, 3, 1:].cpu())
    # plt.show()

    model = SimpleAttention(k=40, n=x.size(1), n_layers=2, hidden_dim=32, dropout=0.1).cuda()
    results = train_model(model, x, y, epochs=1500, lr=5e-3)

    print_matrices_scores(true_attn, **results)


    """"
    x = x[..., -pred.size(-1):]
    y = y[..., -pred.size(-1):]
    true_functions = true_functions[..., -pred.size(-1):]

    for v in range(3):
        plt.plot(y[0, v, :50].cpu(), label='train', color='orange')
        plt.plot(true_functions[0, v, :50].cpu(), label='true', color='blue')
        plt.plot(pred[0, v, :50], label=f'prediction', alpha=0.06, color='blue')
        for i in range(1, len(pred)):
            plt.plot(pred[i, v, :50], alpha=0.06, color='blue')
        plt.title(f'Var {v}')
        plt.legend()
        plt.show()

    plt.scatter(x[0, 0].cpu(), y[0, 1].cpu())
    plt.scatter(x[0, 0].cpu(), pred[0, 1].cpu())
    plt.show()
    plt.scatter(x[0, 1].cpu(), y[0, 2].cpu())
    plt.scatter(x[0, 1].cpu(), pred[0, 2].cpu())
    plt.show()
    plt.scatter(x[0, 0].cpu(), y[0, 2].cpu())
    plt.scatter(x[0, 0].cpu(), pred[0, 2].cpu())
    plt.show()
    plt.scatter(x[0, 0].cpu(), y[0, 0].cpu())
    plt.scatter(x[0, 0].cpu(), pred[0, 0].cpu())
    plt.show()

    plt.scatter(x[0, 0, :-1].cpu(), y[0, 3, 1:].cpu())
    plt.scatter(x[0, 0, :-1].cpu(), pred[0, 3, 1:].cpu())
    plt.show()
    plt.scatter(x[0, 1, :-1].cpu(), y[0, 3, 1:].cpu())
    plt.scatter(x[0, 1, :-1].cpu(), pred[0, 3, 1:].cpu())
    plt.show()
    plt.scatter(x[0, 2, :-1].cpu(), y[0, 3, 1:].cpu())
    plt.scatter(x[0, 2, :-1].cpu(), pred[0, 3, 1:].cpu())
    plt.show()
    """


if __name__ == '__main__':
    torch.set_printoptions(precision=2, sci_mode=False)
    test()
