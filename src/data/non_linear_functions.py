import warnings

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from tqdm import trange

from src.utils.pytorch import fit_regression, find_extrema, fit_regression_model
from src.utils.visualisations import plot_mesh, plot_3d_points

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

    model, additive, score = get_non_linear_functions(adj_matrix, best_of=3, n_points=30, epochs=1500)

    _, _, mesh_data_coupled, points_3d = model.get_mesh_data(precision=30)
    mesh_data_additive = additive.get_mesh_data(precision=30)

    for i in range(len(adj_matrix)):
        ax = plot_mesh(*mesh_data_coupled[i], cmap='viridis')
        ax = plot_mesh(*mesh_data_additive[i], ax=ax, cmap='winter')
        ax = plot_3d_points(*points_3d[i], ax=ax)
        plt.show()


if __name__ == '__main__':
    test_models()
