import torch
import random

from lingam.utils import make_dot
from torch import nn

from src.data.non_linear_functions import get_non_linear_functions


class TemporalCausalGraph(nn.Module):
    """
    Represents a temporal causal graph.

    Args:
        num_nodes (int): The number of nodes in the graph.
        max_lags (int): The maximum number of time lags.

    Attributes:
        causal_matrix (torch.Tensor): The adjacency matrix representing the causal relationships.
        num_nodes (int): The number of nodes in the graph.
        max_lags (int): The maximum number of time lags.
        functional_relationships (NonLinearFunctionSet): The set of non-linear functions representing
                                                         the causal relationships.
        num_external (int): The number of external nodes.
    """

    def __init__(self, num_nodes, max_lags, causal_matrix=None,
                 num_external=0, minimum_incoming_connections=1,
                 external_connections=1, force_max_lags=True):
        super().__init__()
        self.num_nodes = num_nodes
        self.max_lags = max_lags
        self.num_external = num_external
        self.functional_relationships = None
        self.difficulty_score = 0.0

        size = (num_nodes + num_external, num_nodes + num_external, max_lags)
        if causal_matrix is not None:
            assert causal_matrix.shape == size
            self.causal_matrix = causal_matrix
        else:
            self.causal_matrix = torch.zeros(*size, dtype=torch.bool)
            self.connect_graph(
                minimum_incoming_connections=minimum_incoming_connections,
                external_connections=external_connections,
                force_max_lags=force_max_lags
            )

        #self.init_functional_relationships()

    def get_causal_matrix(self,
                          exclude_max_lags=False,
                          exclude_external_incoming=False,
                          exclude_external_outgoing=False,
                          dtype=torch.bool):
        causal_matrix = self.causal_matrix
        if exclude_external_incoming:
            causal_matrix = causal_matrix[:self.num_nodes]
        if exclude_external_outgoing:
            causal_matrix = causal_matrix[:, :self.num_nodes]
        if exclude_max_lags:
            causal_matrix = causal_matrix.any(dim=-1)
        return causal_matrix.to(dtype=dtype)

    def count_edges(self):
        return torch.count_nonzero(self.causal_matrix).item()

    def connect_graph(self, minimum_incoming_connections=1, external_connections=1, force_max_lags=True):
        # Compute the components of the graph
        causal_matrix = self.causal_matrix[:self.num_nodes, :self.num_nodes].any(dim=-1)
        causal_matrix = torch.logical_or(causal_matrix, causal_matrix.t())
        visited = [False] * len(causal_matrix)
        components = []
        for i in range(len(causal_matrix)):
            if not visited[i]:
                current_component = []
                depth_first_search(i, visited, causal_matrix, current_component)
                components.append(current_component)

        random.shuffle(components)
        result = components.pop()
        while len(components) > 0:
            next_component = components.pop()

            edge = [random.choice(result), random.choice(next_component)]
            random.shuffle(edge)

            _to, _from = edge
            _lags = random.randint(0, self.max_lags - 1)
            self.causal_matrix[_to, _from, _lags] = True

            result.extend(next_component)

        # Add minimum number of outgoing external connections
        required_connections = external_connections - self.causal_matrix.sum(dim=(0, 2))[self.num_nodes:]
        for node, amount in zip(range(self.num_nodes, len(self.causal_matrix)), required_connections):
            if amount < 1:
                continue
            edges = torch.nonzero(~self.causal_matrix[:self.num_nodes, node], as_tuple=False).tolist()
            edges = random.sample(edges, k=amount)
            edges = torch.LongTensor(edges).t()  # (2, amount)
            self.causal_matrix[edges[0], node, edges[1]] = True

        # Add minimum number of incoming connections
        required_connections = minimum_incoming_connections - self.causal_matrix.sum(dim=(1, 2))[:self.num_nodes]
        for node, amount in enumerate(required_connections):
            if amount < 1:
                continue
            edges = torch.nonzero(~self.causal_matrix[node, :self.num_nodes], as_tuple=False).tolist()
            edges = random.sample(edges, k=amount)
            edges = torch.LongTensor(edges).t()  # (2, amount)
            self.causal_matrix[node, edges[0], edges[1]] = True

        # Force the maximum amount of lags
        if force_max_lags and not torch.any(self.causal_matrix[..., 0]):
            _to, _from, lags = random.choice(torch.nonzero(self.causal_matrix[:self.num_nodes, :self.num_nodes],
                                                           as_tuple=False).tolist())
            self.causal_matrix[_to, _from, lags] = False
            self.causal_matrix[_to, _from, 0] = True

    def get_edges(self, free_edges=False, exclude_external=False):
        """
        Get the edges in the graph.

        Args:
            free_edges (bool, optional): ..
            exclude_external (bool, optional): ..

        """
        causal_matrix = self.causal_matrix
        if exclude_external:
            causal_matrix = causal_matrix[:self.num_nodes, :self.num_nodes]
        if free_edges:
            causal_matrix = ~causal_matrix
        return torch.nonzero(causal_matrix, as_tuple=False).tolist()

    def init_functional_relationships(self):
        """
        Initialize the functional relationships based on the adjacency matrix.
        """
        coupled, additive, score = get_non_linear_functions(
            causal_matrix=self.causal_matrix,
            best_of=3,
            n_points=30,
            epochs=1500,
            lr=1e-2)
        self.functional_relationships = coupled
        self.difficulty_score = score

    def forward(self, x: torch.Tensor):
        """
        Apply the functional relationships to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of size (n_var, max_lags).

        Returns:
            torch.Tensor: The output tensor of size (n_var,).
        """
        total_nodes = self.num_nodes + self.num_external
        x = x.reshape(1, 1, total_nodes, self.max_lags)
        x = x.expand(-1, total_nodes, -1, -1)  # (1, total_nodes, total_nodes, max_lags)
        return self.functional_relationships(x)[0]

    def plot(self, path=None, view=False):
        """
        Visualize the graph using Graphviz and save it as an image.

        Args:
            path (str, optional): The path to save the image. If not provided, the image will not be saved.
            view (bool, optional): Whether to open the image after saving. Defaults to False.
        """
        matrix = self.causal_matrix.sum(dim=-1)
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-15)
        labels = [f"N{i}" for i in range(self.num_nodes)]
        if self.num_external > 0:
            labels.extend([f"E{i}" for i in range(self.num_external)])
        dot_graph = make_dot(matrix, labels=labels)
        dot_graph.format = 'svg'
        dot_graph.render(path, view=view, cleanup=path is None)


def depth_first_search(node, visited, graph, current_component):
    visited[node] = True
    current_component.append(node)
    for i in range(len(graph)):
        if graph[node][i] and not visited[i]:
            depth_first_search(i, visited, graph, current_component)


if __name__ == '__main__':
    m = TemporalCausalGraph(num_nodes=7, max_lags=16, minimum_incoming_connections=2,
                            num_external=2, external_connections=2)
    m.plot(view=True)

