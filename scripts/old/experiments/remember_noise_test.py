import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from src.models.NAVAR.navar_tcn import NAVAR_TCN
from src.utils.pytorch import count_parameters


model = NAVAR_TCN(n_variables=3, hidden_dim=8, kernel_size=2, n_blocks=3, n_layers_per_block=2)

print('n_parameters per var:', count_parameters(model) // 3)
print('receptive field:', model.receptive_field)

# Generate random synthetic_data
batch_size = 1
sequence_length = 800
data = torch.randn(batch_size, 3, sequence_length + 1)

y = data[..., 1:]
x = data[..., :-1]

if torch.cuda.is_available():
    model = model.cuda()
    x, y = x.cuda(), y.cuda()

# 2. Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 3. Training
epochs = 5000
pbar = trange(epochs)
for epoch in pbar:
    model.train()
    optimizer.zero_grad()

    prediction, _ = model(x)
    loss = criterion(prediction, y)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0 or epoch == epochs - 1:
        pbar.set_description(f"Loss: {loss.item():.4f}")

    if epoch == 1200:
        optimizer = optim.Adam(model.parameters(), lr=0.001)


model.eval()
with torch.no_grad():
    predictions, _ = model(x)
    predictions = predictions.cpu().numpy()

plt.figure(figsize=(6, 3))  # adjust the size for better visualization
for sample_channel in range(3):
    ax = plt.subplot(3, 1, sample_channel+1)  # 3 rows, 1 column, current plot
    plt.plot(np.arange(200, 401, 1), y[0, sample_channel, 200:401].cpu().numpy(), label="Original Data", alpha=0.7)
    plt.plot(np.arange(200, 401, 1), predictions[0, sample_channel, 200:401], label="Model Prediction", linestyle='dashed', alpha=0.7)
    if sample_channel == 0:
        plt.legend()
    if sample_channel != 2:  # if it's not the last plot, hide the x-axis
        plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylabel(f"Var {sample_channel+1}")
    plt.tight_layout()  # to prevent overlap of titles and labels

plt.savefig("C:/Users/mauri/Desktop/thesis_img/memorization.svg", format='svg')
