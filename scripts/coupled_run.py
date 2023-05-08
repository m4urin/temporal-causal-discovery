import torch

from src.data.dataset import toy_data_coupled
from src.models.implementations.navar import NAVAR_SMALL
from src.training.train_model import train_model
from src.utils.visualisations import draw_causal_matrix


def run_experiment():
    ds = toy_data_coupled(time_steps=1000)
    ds2 = ds.abs()
    ds = torch.cat((ds, torch.log(ds2 + 1)), dim=1)
    print(ds.size())
    model, _loss, _, result = train_model(NAVAR_SMALL, data=ds, epochs=2000, val_proportion=0.0, learning_rate=1e-4,
                                          lambda1=0.7, dropout=0.2, weight_decay=1e-4, kernel_size=4, n_layers=1,
                                          hidden_dim=8, show_progress='tqdm', show_plot=False)
    return result[1].matrix(dim=(0, 1)), _loss


pos = None
cm, loss = run_experiment()
print(loss)
threshold = 0.02
pos = draw_causal_matrix(cm[:4, :4], draw_weights=True, threshold=threshold,
                         save_fig=f'plots/coupled_normal.png',
                         pos=pos, title=f"loss={round(loss, 3)}")
pos = draw_causal_matrix(cm[4:, :4], draw_weights=True, threshold=threshold,
                         save_fig=f'plots/coupled_log.png',
                         pos=pos, title=f"loss={round(loss, 3)}")

cm_alt = cm[:4, :4] + cm[4:, :4]
pos = draw_causal_matrix(cm_alt, draw_weights=True, threshold=threshold,
                         save_fig=f'plots/coupled_alt.png',
                         pos=pos, title=f"loss={round(loss, 3)}")
