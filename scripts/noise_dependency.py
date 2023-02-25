from src.data import toy_data_chain_noise
from src.models.navar_small import NAVAR_SMALL
from src.train_model import train_model

ds, true_contributions = toy_data_chain_noise(noise=0.0, time_steps=4000)
true_contributions = true_contributions.squeeze(dim=0)
ds = ds.cuda()
model, loss, _, result = train_model(NAVAR_SMALL, data=ds, epochs=3000, val_proportion=0.0, learning_rate=1e-5,
                                     lambda1=0.1, dropout=0.2, weight_decay=1e-4, kernel_size=3, n_layers=1,
                                     hidden_dim=4, show_progress='tqdm', show_plot=True)
predictions, contributions = result
contributions = contributions.mean(dim=(0, 1))
print(contributions)
print(true_contributions)
