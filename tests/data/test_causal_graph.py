import unittest
import torch
from src.data.causal_graph import TemporalCausalGraph, add_external_variables, generate_causal_matrix


class TemporalCausalGraphTests(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 10
        self.max_lags = 5
        self.graph = TemporalCausalGraph(self.num_nodes, self.max_lags)

    def test_get_num_edges(self):
        self.assertEqual(self.graph.get_num_edges(), 0)
        self.graph.add(0, 1, 0)
        self.assertEqual(self.graph.get_num_edges(), 1)
        self.graph.add(2, 3, 1)
        self.assertEqual(self.graph.get_num_edges(), 2)

    def test_get_num_self_edges(self):
        self.assertEqual(self.graph.get_num_self_edges(), 0)
        self.graph.add(0, 0, 0)
        self.assertEqual(self.graph.get_num_self_edges(), 1)
        self.graph.add(1, 1, 1)
        self.assertEqual(self.graph.get_num_self_edges(), 2)

    def test_add(self):
        self.assertEqual(self.graph.causal_matrix.sum(), 0)
        self.graph.add(0, 1, 0)
        self.assertEqual(self.graph.causal_matrix[0, 1, 0], 1)
        self.graph.add(2, 3, 1, 0.5)
        self.assertEqual(self.graph.causal_matrix[2, 3, 1], 0.5)

    def test_add_random(self):
        weights = [0.2, 0.4, 0.6]
        self.graph.add_random(weights)
        adj_weights = self.graph.causal_matrix[self.graph.causal_matrix > 0].tolist()
        self.assertTrue(torch.allclose(torch.tensor(weights), torch.tensor(sorted(adj_weights))))

    def test_at_least_one_incoming(self):
        self.assertEqual(self.graph.causal_matrix.sum(), 0)
        self.graph.at_least_one_incoming()
        self.assertEqual(self.graph.causal_matrix.sum(), self.num_nodes)

    def test_normalize(self):
        self.graph.add(0, 1, 0)
        self.graph.add(0, 2, 1)
        self.graph.add(0, 3, 2)
        self.graph.normalize()
        self.assertAlmostEqual(self.graph.causal_matrix[0, 1, 0], 1 / 3)
        self.assertAlmostEqual(self.graph.causal_matrix[0, 2, 1], 1 / 3)
        self.assertAlmostEqual(self.graph.causal_matrix[0, 3, 2], 1 / 3)

    def test_get_edges(self):
        self.graph.add(0, 1, 0)
        self.graph.add(2, 3, 1)
        self.assertEqual(self.graph.get_edges(), [(0, 1, 0), (2, 3, 1)])
        self.assertEqual(self.graph.get_edges(return_weights=True), [(0, 1, 0, 1.0), (2, 3, 1, 1.0)])

    def test_get_free_edges(self):
        total = torch.numel(self.graph.causal_matrix)
        self.graph.add(0, 1, 0)
        self.graph.add(2, 3, 1)
        self.assertEqual(len(self.graph.get_free_edges()), total - 2)
        self.assertEqual(len(self.graph.get_free_edges(return_weights=True)), total - 2)
        self.graph.add(4, 5, 0)
        self.assertTrue((4, 5, 0) not in self.graph.get_free_edges())

    def test_init_functional_relationships(self):
        self.graph.add(0, 1, 0)
        self.graph.add(0, 2, 1)
        self.graph.add(0, 3, 2)
        self.graph.init_functional_relationships()
        self.assertIsNotNone(self.graph.functional_relationships)

    def test_forward(self):
        self.graph.init_functional_relationships()
        x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
        output = self.graph.forward(x)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, self.num_nodes))

    def test_add_external_variables(self):
        external_graph = add_external_variables(self.graph, 2, 1)
        self.assertEqual(external_graph.num_nodes, self.num_nodes + 2)
        self.assertEqual(external_graph.num_external, 2)

    def test_generate_causal_matrix(self):
        causal_graph = generate_causal_matrix(self.num_nodes, 10, max_lags=3, min_weight=0.2, num_ext_nodes=2, num_etx_connections=1)
        self.assertEqual(causal_graph.num_nodes, self.num_nodes + 2)
        self.assertEqual(causal_graph.max_lags, 3)
        self.assertEqual(causal_graph.num_external, 2)


if __name__ == '__main__':
    unittest.main()
