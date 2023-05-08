import math
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from definitions import RESULTS_DIR
from src.utils.io import join_paths


def draw_causal_matrix(causal_matrix, threshold=0.02, draw_weights=True, save_fig=None, pos=None, title=None):
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
    causal_matrix = np.where(causal_matrix >= threshold, causal_matrix, 0.0)

    # create multi graph
    G = nx.MultiDiGraph()

    # add nodes
    G.add_nodes_from(list(range(len(causal_matrix))))

    # add edges
    edges = []
    for i in range(len(causal_matrix)):
        for j in range(len(causal_matrix)):
            if causal_matrix[i, j] > 0.01:
                edges.append((i, j, causal_matrix[i, j]))
    G.add_weighted_edges_from(edges)

    # set position
    if pos is None:
        pos = nx.spring_layout(G, seed=0, iterations=50, k=3 / math.sqrt(len(causal_matrix)))

    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=350)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=14)

    # draw parallel edges
    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], width=edge[2]*2.5, connectionstyle='arc3, rad = 0.08', arrowsize=12)

    # draw weights
    if draw_weights:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): str(round(w, 2)) for u, v, w in edges if u != v},
                                     font_size=8)

    if title is not None:
        plt.title(title)

    if save_fig is not None:
        plt.savefig(join_paths(RESULTS_DIR, save_fig))
    else:
        plt.show()
    plt.clf()

    return pos


def plot_multiple_timeseries(data, title=None, names=None, x_label=None):
    plt.clf()
    k = len(data)
    fig, axs = plt.subplots(k, 1, figsize=(8, 6), sharex=True)
    for i in range(k):
        axs[i].plot(data[i])
        if names is not None:
            axs[i].set_ylabel(names[i])
    # Set the x-axis label for the bottom subplot
    axs[-1].set_xlabel("Time" if x_label is None else x_label)

    # Add a title to the figure
    if title is not None:
        fig.suptitle(title)

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.2)
    # Show the plot
    plt.show()


def plot_train_val_loss(train_losses: List[float], val_losses: List[float] = None,
                        test_every: int = 1, path=None, show_plot=False):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train loss')
    if len(val_losses) > 0:
        ax.plot(np.arange(0, len(val_losses)) * test_every, val_losses, label='eval loss')
        ax.set_title('Train and Validation Loss')
    else:
        ax.set_title('Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train and Validation Loss')
    ax.legend()

    if path is not None:
        plt.savefig(path)

    if show_plot:
        plt.show()

    plt.clf()


if __name__ == '__main__':
    _data = [[0, 1, 2, 3, 2, 3, 2], [6, 6, 6, 3, 3, 3, 6, 6, 6], [[1, 2], [4, 4], [8, 6], [9, 7]]]
    plot_multiple_timeseries(_data, names=['cool', 'plot', 'dude'], x_label='years', title="test")



    #my_causal_matrix = [[0.1, 0.0, 0.002, 0.5], [0.9, 0.0, 0.9, 0.0], [0.0, 0.4, 0.1, 0.3], [0.0, 0.1, 0.0, 0.7]]
    #draw_causal_matrix(my_causal_matrix, draw_weights=False, threshold=0.0, save_fig='plots/test_graph.png')
