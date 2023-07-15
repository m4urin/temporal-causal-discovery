import torch

from definitions import DEVICE
from src.models.implementations.navar_tcn import NAVAR_TCN
from src.experiments.train_model import train_model
from src.utils2 import count_parameters
from src.utils.visualisations import draw_causal_matrix, plot_multiple_timeseries


def run_experiment():
    ds = torch.randn(1, 3, 1000, device=DEVICE)
    model, _loss, _epochs, result = train_model(NAVAR_TCN, data=ds, epochs=6000, val_proportion=0.0, learning_rate=1e-4,
                                          lambda1=0.2, dropout=0.0, weight_decay=1e-6, kernel_size=2, n_layers=4,
                                          hidden_dim=5, show_progress='tqdm', show_plot=True)
    print("number_of_parameters:", count_parameters(model))
    print("loss:", _loss)

    true_data = ds[0, :, 1:].cpu()  # (3, 499)
    pred_data = result[0].matrix(dim=(0, 1))[:-1].t().cpu()  # (3, 499)
    plot_data = torch.stack((true_data, pred_data), dim=-1)
    plot_multiple_timeseries(plot_data, title="Remember noise", y_labels=[f"Var {j + 1}" for j in range(len(plot_data))])

    return result[1].matrix(dim=(0, 1)), _loss


pos = None
_contributions, loss = run_experiment()
pos = draw_causal_matrix(_contributions, draw_weights=True, threshold=0.1,
                         save_fig=f'plots/random/contributions.png',
                         pos=pos, title=f"loss={round(loss, 3)}")
