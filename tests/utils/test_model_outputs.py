import unittest
import torch
from src.utils.model_outputs import *


class TestModelOutputs(unittest.TestCase):
    def setUp(self):
        self.predictions = torch.randn(3, 3)

    def test_model_output(self):
        loss = torch.nn.functional.mse_loss(self.predictions, torch.zeros_like(self.predictions))
        model_output = ModelOutput(self.predictions, loss)
        self.assertIs(model_output.predictions, self.predictions)
        self.assertIs(model_output.losses, loss)

    def test_causal_matrix(self):
        mu = torch.randn(3, 3)
        std = torch.randn_like(mu)
        causal_matrix = CausalMatrix(mu, std)
        self.assertIsNotNone(causal_matrix.matrix)
        self.assertIsNotNone(causal_matrix.std)
        self.assertFalse(causal_matrix.is_history_dependent)
        self.assertTrue(causal_matrix.has_std)

    def test_causal_matrix_no_std(self):
        mu = torch.randn(3, 3)
        causal_matrix = CausalMatrix(mu)
        self.assertIsNotNone(causal_matrix.matrix)
        self.assertIsNone(causal_matrix.std)
        self.assertFalse(causal_matrix.is_history_dependent)
        self.assertFalse(causal_matrix.has_std)

    def test_causal_matrix_with_invalid_std(self):
        mu = torch.randn(3, 3)
        std = torch.randn(2, 2)
        with self.assertRaises(AssertionError):
            causal_matrix = CausalMatrix(mu, std)

    def test_eval_output(self):
        train_loss = 0.5
        test_loss = 0.7
        causal_matrices = {'causal_matrix_1': CausalMatrix(torch.randn(3, 3))}
        eval_output = EvalOutput(self.predictions, train_loss, test_loss, causal_matrices)
        self.assertIs(eval_output.predictions, self.predictions)
        self.assertAlmostEqual(eval_output.train_losses, train_loss, places=7)
        self.assertAlmostEqual(eval_output.test_losses, test_loss, places=7)
        self.assertDictEqual(eval_output.causal_matrices, causal_matrices)


if __name__ == '__main__':
    unittest.main()
