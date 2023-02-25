from time import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from src.data import get_causeme_data



experiment = 'TestWEATH_N-10_T-2000'

data = get_causeme_data(f'{experiment}.zip')[0]
print('data', data.size())

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



"""
model = NAVARTCN(10, 3, 1, 16, 0.2)
n_params_navar = sum(p.numel() for p in model.parameters() if p.requires_grad)
t0 = time()
loss_hist_navar, loss_navar = train_tcn_model(model, lambda1=0.15)
print(f'NAVAR, time={round((time()-t0)/60, 1)}min, val_loss={loss_navar}, train_loss={min(loss_hist_navar)}, n_params={n_params_navar}')
"""

model = TemporalConvNet(10, [16, 32, 64], 10, 3, 0.2)
print('receptive field', receptive_field(3, 2))
n_params_tcn = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('n_parameters', n_params_tcn)
t0 = time()
loss_hist_tcn, loss_hist_test_tcn = train_tcn_model(model)
print(f'TCN, time={round((time()-t0)/60, 1)}min, val_loss={round(min(loss_hist_test_tcn), 3)}, train_loss={round(min(loss_hist_tcn), 3)}, n_params={n_params_tcn}')


plt.plot(loss_hist_tcn, label='train')
plt.plot(np.arange(len(loss_hist_test_tcn)) * 100, loss_hist_test_tcn, label='test')
plt.legend()
plt.show()
