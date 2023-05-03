import unittest
import torch

from src.utils.pytorch import *


class TestModelParameterSharing(unittest.TestCase):
    def setUp(self):
        # Define a simple PyTorch model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 30)

        # Create two instances of the SimpleModel class
        self.model1 = SimpleModel()
        nn.init.zeros_(self.model1.fc1.weight.data)
        self.model2 = SimpleModel()

    def test_count_parameters(self):
        num_params1 = count_parameters(self.model1)
        num_params2 = count_parameters(self.model2)

        self.assertEqual(num_params1, (10 * 20) + 20 + (20 * 30) + 30)
        self.assertEqual(num_params1, num_params2)

    def test_share_parameters(self):
        share_parameters(self.model1, self.model2)

        # Count the number of parameters in each model after sharing parameters
        new_num_params1 = count_parameters(self.model1)
        new_num_params2 = count_parameters(self.model2)

        # Assert that the number of parameters is the same before and after sharing parameters
        self.assertEqual(new_num_params1, new_num_params2)

        # Assert that the shared parameters have the same values in both models
        for p1, p2 in zip(self.model1.parameters(), self.model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    unittest.main()
