import torch

from src.data.dataset import toy_data_chain_noise
from src.models.implementations.navar import NAVAR_SMALL
from src.training.train_model import train_model
from src.utils.visualisations import draw_causal_matrix


def run_experiment(noise : float):
    ds, true_contributions = toy_data_chain_noise(noise=noise, time_steps=4000)
    true_contributions = true_contributions.squeeze(dim=0)
    ds = ds.cuda()
    model, _loss, _, result = train_model(NAVAR_SMALL, data=ds, epochs=2500, val_proportion=0.0, learning_rate=1e-4,
                                          lambda1=0.3, dropout=0.2, weight_decay=1e-4, kernel_size=3, n_layers=1,
                                          hidden_dim=4, show_progress='tqdm', show_plot=False)
    return result[1].matrix(dim=(0, 1)), _loss


pos = None
for noise in torch.arange(1.10, 2.0, 0.2):
    cm, loss = run_experiment(noise)
    pos = draw_causal_matrix(cm, draw_weights=True, threshold=0.1,
                             save_fig=f'plots/noise/noise_{round(noise.item(), 1)}.png',
                             pos=pos, title=f"loss={round(loss, 3)}, noise={round(noise.item(), 2)}")
