import math
import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from definitions import RESULTS_DIR
from src.utils import join


def draw_causal_matrix(causal_matrix, treshold=0.02, draw_weights=True, save_fig=None):
    # to np array
    causal_matrix = np.array(causal_matrix, dtype=np.float32)
    # check data
    if causal_matrix.ndim == 1:
        num_nodes = int(math.sqrt(len(causal_matrix)))
        causal_matrix = causal_matrix.reshape(num_nodes, num_nodes)
    assert causal_matrix.ndim == 2 and causal_matrix.shape[0] == causal_matrix.shape[1], \
        "Requires a square causal matrix"

    # scale to 1.0
    max_val = causal_matrix.max()
    if max_val > 0:
        causal_matrix /= causal_matrix.max()
    # apply threshold
    causal_matrix = np.where(causal_matrix >= treshold, causal_matrix, 0.0) * 2.5

    # create multi graph
    G = nx.MultiDiGraph()

    # add nodes
    G.add_nodes_from(list(range(len(causal_matrix))))

    # add edges
    edges = []
    for i in range(len(causal_matrix)):
        for j in range(len(causal_matrix)):
            if causal_matrix[i, j] > 0:
                edges.append((i, j, 0.01 + causal_matrix[i, j]))
    G.add_weighted_edges_from(edges)

    # set position
    pos = nx.spring_layout(G, seed=0, iterations=50, k=3 / math.sqrt(len(causal_matrix)))

    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=350)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=14)

    # draw parallel edges
    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2], connectionstyle='arc3, rad = 0.08', arrowsize=12)

    # draw weights
    if draw_weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): str(round(w, 2)) for u, v, w in edges if u != v},
                                     font_size=8)

    if save_fig is not None:
        plt.savefig(join(RESULTS_DIR, save_fig))
    else:
        plt.show()
    plt.clf()


if __name__ == '__main__':
    my_causal_matrix = [[0.1, 0.0, 0.002, 0.5], [0.9, 0.0, 0.9, 0.0], [0.0, 0.4, 0.1, 0.3], [0.0, 0.1, 0.0, 0.7]]
    draw_causal_matrix(my_causal_matrix, draw_weights=False, treshold=0.0, save_fig='plots/test_graph.png')
