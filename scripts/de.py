import torch
from matplotlib import pyplot as plt

from src.data import cos_sin_features, mackey_glass



x = torch.cat((mackey_glass(500), cos_sin_features(500)), dim=0)


plt.plot(x.t(), label='data')
plt.legend()
plt.show()
