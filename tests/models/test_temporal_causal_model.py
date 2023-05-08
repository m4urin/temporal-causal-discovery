import unittest
import torch

from src.models.temporal_causal_model import *
from src.utils.model_outputs import *


class TestTemporalCausalModel(unittest.TestCase):
    def setUp(self):
        self.model = TemporalCausalModel()

    def test_forward(self):
        # Test that the forward method raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.model(torch.rand(10, 3))

    def test_get_receptive_field(self):
        # Test that the get_receptive_field method raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.model.get_receptive_field()

    def test_get_n_parameters(self):
        # Test that the get_n_parameters method returns the correct number of parameters
        self.assertEqual(self.model.get_n_parameters(), 0)

        # Test that the get_n_parameters method returns the correct number of trainable parameters
        self.assertEqual(self.model.get_n_parameters(trainable_only=True), 0)

    def test_evaluate(self):
        # Test that the evaluate method raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            self.model.evaluate(torch.rand(10, 3))


class TestTemporalCausalModelImplemented(unittest.TestCase):
    def setUp(self):
        class MyTemporalCausalModel(TemporalCausalModel):
            def __init__(self):
                super().__init__()
                # Define the layers of the model
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)

            def forward(self, x: torch.Tensor) -> ModelOutput:
                # Compute the forward pass of the model
                x = self.fc1(x)
                x = self.fc2(x)
                predictions = x.squeeze()
                loss = torch.mean(predictions)
                return ModelOutput(predictions=predictions, losses=loss)

            def get_receptive_field(self) -> int:
                # Return the receptive field of the model
                return 1

            def evaluate(self, x: torch.Tensor) -> EvalOutput:
                # Evaluate the model on the input tensor x and return the evaluation output
                model_output = self.forward(x)
                train_loss = model_output.losses.item()
                test_loss = train_loss + 0.1  # Just for demonstration purposes
                causal_matrices = {'causal_matrix': CausalMatrix(torch.randn(3, 3), torch.randn(3, 3))}
                return EvalOutput(predictions=model_output.predictions, train_loss=train_loss, test_loss=test_loss,
                                  causal_matrices=causal_matrices)
        # Create a temporal causal model for testing
        self.model = MyTemporalCausalModel()

    def test_forward(self):
        # Test that the forward method returns a ModelOutput object
        x = torch.randn(2, 10)
        output = self.model.forward(x)
        self.assertIsInstance(output, ModelOutput)

    def test_receptive_field(self):
        # Test that the get_receptive_field method returns an integer
        receptive_field = self.model.get_receptive_field()
        self.assertIsInstance(receptive_field, int)

    def test_n_parameters(self):
        # Test that the get_n_parameters method returns a non-negative integer
        n_params = self.model.get_n_parameters()
        self.assertIsInstance(n_params, int)
        self.assertGreaterEqual(n_params, 0)

    def test_evaluate(self):
        # Test that the evaluate method returns an EvalOutput object
        x = torch.randn(2, 10)
        eval_output = self.model.evaluate(x)
        self.assertIsInstance(eval_output, EvalOutput)
