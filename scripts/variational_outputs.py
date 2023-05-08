from matplotlib import pyplot as plt

from definitions import DEVICE
from src.data.dataset import toy_data_5_nodes_variational
from src.models.implementations.navar_variational import NAVAR_Variational
from src.training.train_model import train_model

print(DEVICE)
#DEVICE = 'cpu'

data = toy_data_5_nodes_variational()
_, min_loss, best_epochs, result = train_model(NAVAR_Variational, data, epochs=2000,
                                                   val_proportion=0.0, learning_rate=1e-4,
                                                   lambda1=2, dropout=0.2, weight_decay=1e-3,
                                                   kernel_size=4, n_layers=2, hidden_dim=5,
                                                   show_progress='tqdm', show_plot=True)

predictions, mu, std = result
predictions = predictions[..., :-1, 0].cpu()

epistemic_mean = predictions.matrix(dim=0).flatten()
epistemic_std = predictions.std(dim=0).flatten()
data = data[:, 0, 4:].flatten().cpu() / 4

print(predictions.size())
print(epistemic_mean.size())
print(epistemic_std.size())
print(data.size())


plt.plot(data, label='true')
plt.plot(epistemic_mean, label='epistemic_mean')
plt.plot(epistemic_mean + 2 * epistemic_std, label='+2xstd')
plt.plot(epistemic_mean - 2 * epistemic_std, label='-2xstd')
plt.legend()
plt.show()

print(mu.size())
print(std.size())
