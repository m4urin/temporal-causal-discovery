import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot
from matplotlib import pyplot as plt
import networkx as nx

print([np.__version__, pd.__version__, graphviz.__version__, lingam.__version__])

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)

x3 = np.random.uniform(size=10000)
x0 = 3.0*x3 + np.random.uniform(size=10000)
x2 = 6.0*x3 + np.random.uniform(size=10000)
x1 = 3.0*x0 + 2.0*x2 + np.random.uniform(size=10000)
x5 = 4.0*x0 + np.random.uniform(size=10000)
x4 = 8.0*x0 - 1.0*x2 + np.random.uniform(size=10000)
X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])


model = lingam.DirectLiNGAM()
model.fit(X)
print(model.adjacency_matrix_)
model.adjacency_matrix_[3, 5] = 1.0

dot = make_dot(model.adjacency_matrix_)

dot.format = 'png'
dot.render('result')


