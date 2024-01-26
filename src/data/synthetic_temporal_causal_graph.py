import random
import torch
from torch import nn
from tqdm import trange

from src.data.non_linear_relationships import get_non_linear_functions


class SyntheticTCG(nn.Module):
    """
    A synthetic temporal causal graph generator that creates a graph with specified properties.
    It includes functional relationships, and connectivity constraints.
    """
    def __init__(self, num_internal_nodes, max_lags, min_incoming_connections=1,
                 num_external_nodes=0, num_ext_connections=1):
        super().__init__()
        self.num_internal_nodes = num_internal_nodes
        self.num_external_nodes = num_external_nodes
        self.num_total_nodes = num_internal_nodes + num_external_nodes
        self.max_lags = max_lags
        self.min_incoming_connections = min_incoming_connections
        self.num_ext_connections = num_ext_connections

        self.functional_relationships = None
        self.difficulty_score = 0.0

        total_nodes = self.num_total_nodes
        self.causal_matrix = torch.zeros(total_nodes, total_nodes, max_lags, dtype=torch.bool)
        self.connect_graph(min_incoming_connections, num_ext_connections)

    def permute_random_connection(self):
        """
        Permute a random connection within the causal matrix of the temporal causal graph.
        This operation creates a new causal matrix with a connection added and another removed,
        then constructs a new graph with the modified matrix. Note that functional relationships are not cloned.
        """
        new_causal_matrix = self.causal_matrix.clone()
        a1, a2, a3 = random.choice(torch.nonzero(new_causal_matrix)).tolist()  # indices to add
        #r1, r2, r3 = random.choice(torch.nonzero(new_causal_matrix)).tolist()  # indices to remove
        new_causal_matrix[a1, a2, a3] = False
        #new_causal_matrix[r1, r2, r3] = False

        new_graph = SyntheticTCG(self.num_internal_nodes, self.max_lags, self.min_incoming_connections,
                                 self.num_external_nodes, self.num_external_nodes)
        new_graph.causal_matrix = new_causal_matrix
        new_graph.functional_relationships = self.functional_relationships.replace_causal_matrix(new_causal_matrix)
        return new_graph

    def get_causal_matrix(self):
        """
        Retrieve the causal matrix of the synthetic temporal causal graph by collapsing it along the lag dimension
        and converting it to a float tensor.
        """
        return self.causal_matrix.any(dim=-1).float()

    def get_num_edges(self):
        return torch.count_nonzero(self.causal_matrix).item()

    def connect_graph(self, min_incoming_connections=1, num_ext_connections=1):
        """
        Connect the synthetic temporal causal graph by generating a set of components, ensuring connections
        between them. The graph construction includes the addition of incoming and outgoing connections to
        meet specified requirements, as well as forcing the maximum amount of lags for connections.
        """
        # Compute the components of the graph
        causal_matrix = self.causal_matrix[:self.num_internal_nodes, :self.num_internal_nodes].any(dim=-1)
        causal_matrix = torch.logical_or(causal_matrix, causal_matrix.t())
        visited = [False] * len(causal_matrix)
        components = []
        for i in range(len(causal_matrix)):
            if not visited[i]:
                current_component = []
                _depth_first_search(i, visited, causal_matrix, current_component)
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
        required_connections = num_ext_connections - self.causal_matrix.sum(dim=(0, 2))[self.num_internal_nodes:]
        for node, amount in zip(range(self.num_internal_nodes, len(self.causal_matrix)), required_connections):
            if amount < 1:
                continue
            edges = torch.nonzero(~self.causal_matrix[:self.num_internal_nodes, node], as_tuple=False).tolist()
            edges = random.sample(edges, k=amount)
            edges = torch.LongTensor(edges).t()  # (2, amount)
            self.causal_matrix[edges[0], node, edges[1]] = True

        # Add minimum number of incoming connections
        required_connections = min_incoming_connections - self.causal_matrix.sum(dim=(1, 2))[:self.num_internal_nodes]
        for node, amount in enumerate(required_connections):
            if amount < 1:
                continue
            edges = torch.nonzero(~self.causal_matrix[node, :self.num_internal_nodes], as_tuple=False).tolist()
            edges = random.sample(edges, k=amount)
            edges = torch.LongTensor(edges).t()  # (2, amount)
            self.causal_matrix[node, edges[0], edges[1]] = True

        # Force the maximum amount of lags
        if not torch.any(self.causal_matrix[..., 0]):
            _to, _from, lags = random.choice(torch.nonzero(self.causal_matrix[:self.num_internal_nodes, :self.num_internal_nodes],
                                                           as_tuple=False).tolist())
            self.causal_matrix[_to, _from, lags] = False
            self.causal_matrix[_to, _from, 0] = True

    def init_functional_relationships(self):
        """ Initialize the functional relationships based on the adjacency matrix. """
        coupled, _, score = get_non_linear_functions(
            causal_matrix=self.causal_matrix, best_of=1, n_points=30, epochs=1000, lr=5e-3)
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
        total_nodes = self.num_internal_nodes + self.num_external_nodes
        x = x.reshape(1, 1, total_nodes, self.max_lags)
        x = x.expand(-1, total_nodes, -1, -1)  # (1, total_nodes, total_nodes, max_lags)
        return self.functional_relationships(x)[0]


def generate_causal_graph(num_internal_nodes, max_lags, min_incoming_connections=1,
                          num_external_nodes=0, num_ext_connections=1):
    """
    Generate a SyntheticTCG object with specified properties.
    Args:
        num_internal_nodes (int): Number of internal nodes in the graph.
        max_lags (int): Maximum number of time lags.
        min_incoming_connections (int): Minimum incoming connections per node.
        num_external_nodes (int): Number of external nodes in the graph.
        num_ext_connections (int): Number of connections for external nodes.
    Returns:
        SyntheticTCG: An instance of the SyntheticTCG class with the specified properties.
    """
    synthetic_tcg = SyntheticTCG(num_internal_nodes, max_lags, min_incoming_connections,
                                 num_external_nodes, num_ext_connections)
    return synthetic_tcg


def generate_data(contemporaneous_milestones: list[tuple[int, SyntheticTCG]], sequence_length: int = 500,
                  warmup: int = 200, noise_factor: float = 0.4):
    """
    Construct contemporaneous data using a series of synthetic causal graphs at specific milestones.
    The function generates random data with noise and incorporates the effects of the causal relationships
    defined by the graphs. The resulting data, data mean, and ground truth are returned as tensors.
    """
    assert sequence_length > 0 and warmup >= 0, \
        "Invalid arguments: time_steps, and warmup must be positive integers"

    internal_nodes = contemporaneous_milestones[0][1].num_internal_nodes
    external_nodes = contemporaneous_milestones[0][1].num_external_nodes
    assert all(internal_nodes == cg.num_internal_nodes and
               external_nodes == cg.num_external_nodes for _, cg in contemporaneous_milestones), \
        'requires all the graphs to have the same number of internal/external nodes'

    total_nodes = internal_nodes + external_nodes
    max_lags = max(cg.max_lags for _, cg in contemporaneous_milestones)

    # Generate random data
    data = noise_factor * torch.randn(total_nodes, max_lags + warmup + sequence_length + 1,
                                      device='cuda' if torch.cuda.is_available() else 'cpu')
    data_mean = torch.zeros_like(data)
    gt = generate_ground_truth(contemporaneous_milestones, sequence_length)

    offset = data.size(-1) - sequence_length

    contemporaneous_milestones = list(contemporaneous_milestones)
    with torch.no_grad():
        _, current_graph = contemporaneous_milestones.pop(0)
        for i in trange(current_graph.max_lags, data.size(-1), desc='Generating temporal data..'):
            if len(contemporaneous_milestones) > 0 and contemporaneous_milestones[0][0] == i - offset:
                _, current_graph = contemporaneous_milestones.pop(0)
            effect_without_noise = current_graph.forward(data[:, i - current_graph.max_lags:i])
            data_mean[:, i] = effect_without_noise
            data[:, i] += effect_without_noise

    # add batch_size and remove dependency on sliced tensor
    data = data[None, :internal_nodes, -sequence_length:].cpu().clone()
    data_mean = data_mean[None, :internal_nodes, -sequence_length:].cpu().clone()

    return data, data_mean, gt


def generate_ground_truth(causal_graphs_milestones, sequence_length):
    """
    Generate ground truth data for the given causal graph milestones and sequence length.
    The ground truth is constructed by replicating the causal matrix of each graph and expanding
    it across the sequence length. The resulting tensor represents the causal relationships for
    each time step and graph milestone.
    """
    milestone_sizes = _verify_milestones([i for i, _ in causal_graphs_milestones], sequence_length)
    gt = []
    for size_, (_, cg) in zip(milestone_sizes, causal_graphs_milestones):
        causal_matrix = cg.get_causal_matrix()[:cg.num_internal_nodes, :, None]
        causal_matrix = causal_matrix.expand(-1, -1, size_)
        gt.append(causal_matrix)
    gt = torch.cat(gt, dim=-1).unsqueeze(0).cpu().clone()
    return gt


def _verify_milestones(mile_stones: list[int], sequence_length: int):
    if len(mile_stones) == 0:
        raise ValueError("Milestones list is empty.")

    if mile_stones[0] != 0:
        raise ValueError("First milestone should be 0.")

    if mile_stones[-1] >= sequence_length:
        raise ValueError("Last milestone should be smaller than sequence length.")

    for i in range(1, len(mile_stones)):
        if mile_stones[i] <= mile_stones[i - 1]:
            raise ValueError("Milestones should only increase.")

    milestone_diffs = [mile_stones[i] - mile_stones[i - 1] for i in range(1, len(mile_stones))]
    milestone_diffs.append(sequence_length - mile_stones[-1])
    return milestone_diffs


def _depth_first_search(node, visited, graph, current_component):
    """ A depth-first search on a graph starting from a given node. """
    visited[node] = True
    current_component.append(node)
    for i in range(len(graph)):
        if graph[node][i] and not visited[i]:
            _depth_first_search(i, visited, graph, current_component)
