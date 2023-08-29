import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from src.models.navar import NAVAR_Aleatoric
from src.utils import count_parameters

model = NAVAR_Aleatoric(n_variables=3, hidden_dim=10, kernel_size=2, n_blocks=2, n_layers_per_block=1)

print('n_parameters per var:', count_parameters(model) // 3)
print('receptive field:', model.receptive_field)

# Generate random data
batch_size = 1
sequence_length = 800
data = torch.randn(batch_size, 3, sequence_length + 1)
#data[0, 1] *= -0.9 * (data[0, 1] < 0).float() + 1
#data[0, 1, 250:1250] *= 0.1
data[0, 1] = data[0, 1] ** 3
true_relationships = torch.zeros_like(data)

true_relationships[0, 1, 1:] = torch.sin(4 * data[0, 0, :-1])
#scale = 0.5 * torch.abs(data[0, 0, :-1]).sqrt()
scale = (1 - torch.tanh(2 * data[0, 0, :-1]) ** 2) + 0.2
noise = scale * torch.randn_like(scale)
data[0, 1, 1:] = noise + true_relationships[0, 1, 1:]

true_relationships[0, 2, 1:] = torch.tanh(-0.7 * data[0, 1, :-1])
scale = 1 - torch.tanh(data[0, 1, :-1]) ** 2
noise = scale * torch.randn_like(scale)
data[0, 2, 1:] = noise + true_relationships[0, 2, 1:]

# normalize
d_mean, d_std = data.mean(dim=-1, keepdim=True), data.std(dim=-1, keepdim=True)
data = (data - d_mean) / d_std
true_relationships = (true_relationships - d_mean) / d_std

y = data[..., 1:]
x = data[..., :-1]

if torch.cuda.is_available():
    model = model.cuda()
    x, y = x.cuda(), y.cuda()

# 2. Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)


def custom_loss(_y_true, _mu, _log_var):
    mse_loss = (_y_true - _mu) ** 2
    uncertainty_term = torch.exp(-_log_var) * mse_loss
    variance_reg = 0.5 * _log_var
    return torch.mean(uncertainty_term + variance_reg)


# 1 / var + 0.5 * log(var)

beta = 1  # regularization term

# 3. Training
epochs = 1500
pbar = trange(epochs)
for epoch in pbar:
    model.train()
    optimizer.zero_grad()

    sampled, mu, log_var, contr = model(x)  # mu is the mean, log_var is the logarithm of variance for stability.
    mse_loss = criterion(sampled, y)
    uncertainty_loss = custom_loss(y, mu, log_var)
    reg_loss = contr.abs().mean()
    loss = 0.1 * mse_loss + 0.45 * uncertainty_loss + 0.45 * reg_loss

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        pbar.set_description(f"Loss: {loss.item():.4f}")

    if epoch == epochs // 2:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

model.eval()
with torch.no_grad():
    _, predictions, log_var, contr = model(x)
    predictions = predictions.cpu().numpy()
    contr = contr.cpu().numpy()
    std = torch.exp(0.5 * log_var)
    std = std.cpu().numpy()

plt.figure(figsize=(6, 4))  # adjust the size for better visualization
for sample_channel in range(3):
    ax = plt.subplot(3, 1, sample_channel + 1)  # 3 rows, 1 column, current plot

    plt.plot(np.arange(200, 301, 1), y[0, sample_channel, 200:301].cpu().numpy(),
             label="Original Data", alpha=1.0, zorder=0, color='coral')

    plt.plot(np.arange(200, 301, 1), true_relationships[0, sample_channel, 201:302],
             label="True Contributions", alpha=0.4, zorder=1, color='purple')

    plt.plot(np.arange(200, 301, 1), predictions[0, sample_channel, 200:301],
             label="Model Prediction (μ)", linestyle='dashed', alpha=0.9, zorder=1, color='royalblue')

    # Plotting standard deviations
    plt.fill_between(np.arange(200, 301, 1),
                     predictions[0, sample_channel, 200:301] - 2 * std[0, sample_channel, 200:301],
                     predictions[0, sample_channel, 200:301] + 2 * std[0, sample_channel, 200:301],
                     color='lightsteelblue', alpha=0.2, zorder=-2)

    # Plotting standard deviations
    plt.fill_between(np.arange(200, 301, 1),
                     predictions[0, sample_channel, 200:301] - std[0, sample_channel, 200:301],
                     predictions[0, sample_channel, 200:301] + std[0, sample_channel, 200:301],
                     color='lightsteelblue', alpha=0.3, label="Aleatoric Uncertainty (2σ)", zorder=-1)

    if sample_channel == 0:
        plt.legend()

    if sample_channel != 2:  # if it's not the last plot, hide the x-axis
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylabel(f"Var {sample_channel + 1}")
    ax.set_ylim([-2.2, 2.2])
    plt.tight_layout()  # to prevent overlap of titles and labels

plt.savefig("C:/Users/mauri/Desktop/thesis_img/memorization_std.svg", format='svg')


#x, y, predictions, std

#x_true = torch.from_numpy(np.linspace(-2, 2, 100))
#y_true = 2 * torch.sin(4 * x_true) / d_std[0, 0]
# Sorting based on x values
x, y = x.cpu().numpy(), y.cpu().numpy()
sort_indices = np.argsort(x[0, 0])
x_sorted = x[0, 0, sort_indices]
y_mu_sorted = predictions[0, 1, sort_indices]
y_std_sorted = std[0, 1, sort_indices]
y_sorted = y[0, 1, sort_indices]
true_relationships_sorted = true_relationships[0, 1, sort_indices+1]
contr_sorted = contr[0, 0, 1, sort_indices]

plt.figure(figsize=(10, 6))

# Plotting the true function
plt.grid(True)


plt.scatter(x_sorted, y_sorted, color='blue', marker='o', s=1, label='Training Data', alpha=0.2)

# Plotting the prediction mean and std

# prediction
plt.plot(x_sorted, y_mu_sorted, label='Prediction Mean', color='green', alpha=0.7)
# contribution
#plt.plot(x_sorted, contr_sorted, label='Contribution', color='purple')

plt.fill_between(x_sorted, y_mu_sorted - y_std_sorted, y_mu_sorted + y_std_sorted, color='green', alpha=0.2, label='Prediction Std.')

plt.plot(x_sorted, true_relationships_sorted, label='True Function', color='blue', alpha=0.6)
plt.xlim([-3, 3])
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

