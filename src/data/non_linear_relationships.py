import os
import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from src.utils import exponential_scheduler_with_warmup, get_module_device, TEST_DIR
from src.eval.visualisations import plot_3d_surface, plot_3d_scatter_points

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore", category=UserWarning, message=r'.*scheduler\.step\(\).*')


class CoupledNonLinearRandomFit(nn.Module):
    def __init__(self, causal_matrix, hidden_dim=16, max_input_value=2.0, max_output_value=1.0):
        super().__init__()
        self.causal_matrix = causal_matrix
        self.hidden_dim = hidden_dim
        self.n_var, _, self.max_lags = causal_matrix.size()

        self.max_input_value = max_input_value
        self.max_output_value = max_output_value

        self.register_buffer('alpha', torch.ones(1, self.n_var))
        self.register_buffer('beta', torch.zeros(1, self.n_var))

        mask = ~causal_matrix.bool().reshape(1, self.n_var, self.n_var, self.max_lags)
        self.register_buffer('mask', mask)

        self.conv1 = nn.Conv2d(in_channels=self.n_var,
                               out_channels=self.n_var * hidden_dim,
                               kernel_size=(self.n_var, self.max_lags),
                               groups=self.n_var)
        self.conv2 = nn.Conv2d(in_channels=self.n_var * hidden_dim,
                               out_channels=self.n_var * hidden_dim,
                               kernel_size=(1, 1),
                               groups=self.n_var)
        self.conv3 = nn.Conv2d(in_channels=self.n_var * hidden_dim,
                               out_channels=self.n_var,
                               kernel_size=(1, 1),
                               groups=self.n_var)

        self.x, self.y = None, None

    def replace_causal_matrix(self, causal_matrix):
        assert self.causal_matrix.shape == causal_matrix.shape
        device = get_module_device(self)
        new_instance = self.__class__(causal_matrix, self.hidden_dim, self.max_input_value, self.max_output_value)
        new_instance.to(device)
        new_instance.load_state_dict(self.state_dict())  # Copy parameters and buffers
        new_instance.causal_matrix = causal_matrix
        mask = ~causal_matrix.bool().reshape(1, self.n_var, self.n_var, self.max_lags).to(device)
        new_instance.register_buffer('mask', mask)
        return new_instance

    def forward(self, x: torch.Tensor):
        # (bs, n_var, n_var, max_lags)
        x = torch.masked_fill(x, self.mask, value=0.0)
        # (bs, n_var, n_var, max_lags)
        x = torch.sigmoid(self.conv1(x))
        # (bs, n_var * hidden_dim, 1, 1)
        x = torch.sigmoid(self.conv2(x))
        # (bs, n_var * hidden_dim, 1, 1)
        x = self.conv3(x)
        # (bs, n_var, 1, 1)
        x = x.reshape(-1, self.n_var)
        # (bs, n_var)
        x = self.alpha * x + self.beta  # scale output to (-1, 1)
        # (bs, n_var)
        return x

    def fit_random(self, n_points=20, epochs=2000, min_max_epochs=800, lr=1e-2, pbar=None, show_loss=False):
        x = torch.rand(n_points, self.n_var, self.n_var, self.max_lags, device=DEVICE)
        x = 2 * self.max_input_value * x - self.max_input_value  # range (-1.5, 1.5)
        y = torch.rand(n_points, self.n_var, device=DEVICE)
        y = 2 * self.max_output_value * y - self.max_output_value  # range (-1.0, 1.0)

        fit_regression(self, x, y, epochs=epochs, lr=lr, weight_decay=1e-8, pbar=pbar, show_loss=show_loss)

        min_v, max_v = find_extrema(self, input_size=(self.n_var, self.n_var, self.max_lags),
                                    k=200, epochs=min_max_epochs, output_size=(self.n_var,),
                                    min_input_val=-self.max_input_value, max_input_val=self.max_input_value, pbar=pbar, show_loss=show_loss)

        alpha = 2 * self.max_output_value / (max_v - min_v).clamp(min=1e-15)
        beta = -alpha * min_v - self.max_output_value
        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)

        y = y * self.alpha + self.beta
        self.x, self.y = x, y

        return self

    def get_mesh_data(self, precision):
        if self.x is None:
            raise Exception("Please call fit_random() first.")
        with torch.no_grad():
            lin_space = np.linspace(-self.max_input_value, self.max_input_value, precision)
            x1, x2 = np.meshgrid(lin_space, lin_space)
            x_pairs = torch.tensor(np.column_stack((x1.flatten(), x2.flatten())), dtype=torch.float32, device='cuda')

            x_tensor = torch.zeros(precision * precision, self.n_var, self.n_var, self.max_lags, device='cuda')
            for i in range(self.n_var):
                incoming, lags = self.causal_matrix[i].nonzero(as_tuple=True)
                x_tensor[:, i, incoming, lags] = x_pairs

            y_tensor = self.forward(x_tensor)  # (precision * precision, n_var)

            y1 = y_tensor.t().reshape(self.n_var, precision, precision).cpu().numpy()
            x1 = np.stack([x1 for _ in range(self.n_var)])
            x2 = np.stack([x2 for _ in range(self.n_var)])
            mesh_data = np.stack((x1, x2, y1), axis=1)  # (n_var, 3, precision, precision)

            # n_points, self.n_var, self.n_var, self.max_lags
            points_3d = []
            for i in range(self.n_var):
                incoming, lags = self.causal_matrix[i].nonzero(as_tuple=True)
                points_3d.append(self.x[:, i, incoming, lags].t())
            x_points = torch.stack(points_3d)  # (var, 2, points)
            y_points = self.y.t().reshape(self.n_var, 1, -1)  # (var, 1, points)
            points_3d = torch.cat((x_points, y_points), dim=1)  # (var, 3, points)
            points_3d = points_3d.cpu().numpy()  # (var, 3, points)

            return x_tensor, y_tensor, mesh_data, points_3d


class AdditiveModel(nn.Module):
    def __init__(self, causal_matrix, hidden_dim=16, max_input_value=2.0, max_output_value=1.0):
        super().__init__()
        self.causal_matrix = causal_matrix
        self.hidden_dim = hidden_dim
        self.n_var, _, self.max_lags = causal_matrix.size()

        self.max_input_value = max_input_value
        self.max_output_value = max_output_value

        mask = ~causal_matrix.bool().reshape(1, self.n_var * self.n_var, self.max_lags)
        self.register_buffer('mask', mask)

        self.conv1 = nn.Conv1d(in_channels=self.n_var * self.n_var,
                               out_channels=self.n_var * self.n_var * hidden_dim,
                               kernel_size=self.max_lags,
                               groups=self.n_var * self.n_var)
        self.conv2 = nn.Conv1d(in_channels=self.n_var * self.n_var * hidden_dim,
                               out_channels=self.n_var * self.n_var * hidden_dim,
                               kernel_size=1,
                               groups=self.n_var * self.n_var)
        self.conv3 = nn.Conv1d(in_channels=self.n_var * self.n_var * hidden_dim,
                               out_channels=self.n_var,
                               kernel_size=1,
                               groups=self.n_var)

    def forward(self, x: torch.Tensor):
        # (bs, n_var, n_var, max_lags)
        x = x.reshape(-1, self.n_var * self.n_var, self.max_lags)
        # (bs, n_var * n_var, max_lags)
        x = torch.masked_fill(x, self.mask, value=0.0)
        # (bs, n_var * n_var, max_lags)
        x = torch.sigmoid(self.conv1(x))
        # (bs, n_var * n_var * hidden_dim, 1)
        x = torch.sigmoid(self.conv2(x))
        # (bs, n_var * n_var * hidden_dim, 1)
        x = self.conv3(x)
        # (bs, n_var, 1)
        x = x.reshape(-1, self.n_var)
        # (bs, n_var)
        return x

    def get_mesh_data(self, precision):
        with torch.no_grad():
            lin_space = np.linspace(-self.max_input_value, self.max_input_value, precision)
            x1, x2 = np.meshgrid(lin_space, lin_space)
            x_pairs = torch.tensor(np.column_stack((x1.flatten(), x2.flatten())), dtype=torch.float32, device='cuda')

            x_tensor = torch.zeros(precision * precision, self.n_var, self.n_var, self.max_lags, device='cuda')
            for i in range(self.n_var):
                incoming, lags = self.causal_matrix[i].nonzero(as_tuple=True)
                x_tensor[:, i, incoming, lags] = x_pairs

            y_tensor = self.forward(x_tensor)  # (precision * precision, n_var)

            y1 = y_tensor.t().reshape(self.n_var, precision, precision).cpu().numpy()
            x1 = np.stack([x1 for _ in range(self.n_var)])
            x2 = np.stack([x2 for _ in range(self.n_var)])
            mesh_data = np.stack((x1, x2, y1), axis=1)  # (n_var, 3, precision, precision)

            return mesh_data


def get_non_linear_functions(causal_matrix, best_of=5, n_points=25, epochs=1000, lr=1e-2):
    assert best_of > 0
    min_max_epochs = epochs // 2
    pbar = trange(best_of * (2 * epochs + min_max_epochs))
    best_coupled, best_additive = None, None
    worst_additive_loss = 0

    for _ in range(best_of):
        model = CoupledNonLinearRandomFit(causal_matrix).to(DEVICE)
        model.fit_random(
            n_points=n_points,
            epochs=epochs,
            min_max_epochs=min_max_epochs,
            lr=lr,
            pbar=pbar
        )
        additive = AdditiveModel(causal_matrix).to(DEVICE)
        loss = fit_regression_model(additive, model, data_size=(512, model.n_var, model.n_var, model.max_lags),
                                    epochs=epochs, lr=lr, pbar=pbar)

        if loss > worst_additive_loss:
            worst_additive_loss = loss
            best_coupled, best_additive = model, additive

    return best_coupled, best_additive, int(1000 * worst_additive_loss)


def test_models():
    adj_matrix = torch.zeros(3, 3, 4)
    adj_matrix[0, 0, 2] = 1
    adj_matrix[0, 1, 3] = 1
    adj_matrix[1, 1, 0] = 1
    adj_matrix[1, 2, 0] = 1
    adj_matrix[2, 2, 1] = 1
    adj_matrix[2, 1, 3] = 1
    adj_matrix = adj_matrix.bool()

    model, additive, score = get_non_linear_functions(adj_matrix, best_of=1, n_points=20, epochs=500)

    _, _, mesh_data_coupled, points_3d = model.get_mesh_data(precision=30)
    mesh_data_additive = additive.get_mesh_data(precision=30)

    for i in range(len(adj_matrix)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = plot_3d_surface(*mesh_data_coupled[i], ax=ax, cmap='twilight', label='Coupled Model',
                             color=np.array([[187.0, 132, 149, 256]])/256)
        ax = plot_3d_surface(*mesh_data_additive[i], ax=ax, cmap='winter', label='Additive Model',
                             color=np.array([[7.0, 148, 168, 256]])/256)
        x1, x2, y = points_3d[i]
        ax = plot_3d_scatter_points(x1, x2, y + 0.02, ax=ax, label='Random generation data')

        plt.legend()
        ax.view_init(azim=60, elev=20)

        fig.savefig(os.path.join(TEST_DIR, f'relationship_{i}.png'), dpi=200)
        plt.show()


def strong_interaction():
    lin_space = np.linspace(-1, 1, 20)
    x1, x2 = np.meshgrid(lin_space, lin_space)
    y = x1 * x2 #(20, 20)

    plot_3d_surface(x1, x2, y, ax=None, cmap=None, label=None, color=None)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = plot_3d_surface(x1, x2, x1*x2, ax=ax, cmap='twilight', label='Non-additive Model',
                         color=np.array([[187.0, 132, 149, 256]]) / 256)
    ax = plot_3d_surface(x1, x2, x1*0, ax=ax, cmap='winter', label='Additive Model',
                         color=np.array([[7.0, 148, 168, 256]]) / 256)

    x1, x2 = 2 * torch.rand(2, 25) - 1
    ax = plot_3d_scatter_points(x1, x2, x1 * x2 + torch.randn(25)*0.03, ax=ax, label='Random generation data')

    ticks = np.arange(-1.0, 1.1, 0.5)  # Creates an array from -2 to 2 with a step of 0.5
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    plt.legend()
    ax.view_init(azim=73, elev=15)

    fig.savefig(os.path.join(TEST_DIR, f'strong_interaction.png'), dpi=400)
    plt.show()


def find_minimum(model, input_size, output_size, k=1000, lr=1e-2, epochs=400,
                 min_input_val=None, max_input_val=None, show_loss=False, pbar=None, find_maximum=False):
    assert len(input_size) > 0

    obj = -1 if find_maximum else 1

    clamped = min_input_val is not None or max_input_val is not None

    model.eval()  # switch model to evaluation mode

    # Track the best minimum and maximum
    best_minimum = torch.full((k, *output_size), torch.inf, device=DEVICE)

    losses = []

    # Initialize a random point in the input space
    x = torch.autograd.Variable(torch.randn(k, *input_size).to(DEVICE), requires_grad=True)

    # Minimize the output with gradient descent
    optimizer = torch.optim.Adam([x], lr=lr)
    lr_scheduler = exponential_scheduler_with_warmup(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=0.01,
        warmup_ratio=0.05,
        cooldown_ratio=0.5,
        total_iters=epochs)

    if pbar is None:
        pbar = trange(epochs, desc="Find min/max..")
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = x
        if clamped:
            z = z.clamp(min=min_input_val, max=max_input_val)
        y = obj * model(z)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        best_minimum = torch.minimum(best_minimum, y.detach().view(k, *output_size, -1).mean(dim=-1))
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(desc=f"Find min/max.. Loss={round(loss.item(), 4)}")
        pbar.update()

    if show_loss:
        plt.clf()
        plt.plot(losses, label='Train loss')
        plt.legend()
        plt.show()

    return obj * best_minimum.min(dim=0, keepdim=True).values


def find_extrema(model, input_size, output_size, k=1000, lr=1e-2, epochs=800,
                 min_input_val=None, max_input_val=None, show_loss=False, pbar=None):
    min_v = find_minimum(model, input_size, output_size, k, lr, epochs // 2,
                         min_input_val, max_input_val, show_loss, pbar,
                         find_maximum=False)
    max_v = find_minimum(model, input_size, output_size, k, lr, epochs - (epochs // 2),
                         min_input_val, max_input_val, show_loss, pbar,
                         find_maximum=True)
    return min_v, max_v


def fit_regression(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs=2000, lr=1e-2,
                   weight_decay=1e-6, show_loss=False, pbar=None):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = exponential_scheduler_with_warmup(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=0.01,
        warmup_ratio=0.05,
        cooldown_ratio=0.5,
        total_iters=epochs)

    loss_fn = nn.MSELoss()
    losses = []

    if pbar is None:
        pbar = trange(epochs, desc='Training..')

    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

        # Print loss for tracking progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(desc=f"Training.. Loss={round(loss.item(), 4)}")
        pbar.update()

    if show_loss:
        plt.clf()
        plt.plot(losses, label='Train loss')
        plt.legend()
        plt.show()

    return min(losses)


def fit_regression_model(model: nn.Module, model_original: nn.Module, data_size, epochs=2000, lr=1e-2,
                         weight_decay=1e-6, show_loss=False, pbar=None):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = exponential_scheduler_with_warmup(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=0.01,
        warmup_ratio=0.05,
        cooldown_ratio=0.5,
        total_iters=epochs)

    loss_fn = nn.MSELoss()
    losses = []

    if pbar is None:
        pbar = trange(epochs, desc='Training..')

    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()
        x = torch.randn(*data_size, device=DEVICE)
        with torch.no_grad():
            y = model_original(x)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

        # Print loss for tracking progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(desc=f"Training.. Loss={round(loss.item(), 4)}")
        pbar.update()

    if show_loss:
        plt.clf()
        plt.plot(losses, label='Train loss')
        plt.legend()
        plt.show()

    return min(losses)


if __name__ == '__main__':
    strong_interaction()
