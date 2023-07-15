import unittest
from torch import rand

from src.models.modules.positional_embedding import *


class TestPositionalEmbedding(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_variables = 3
        self.embedding_dim = 128
        self.seq_length = 300
        self.pe = PositionalEmbedding(num_variables=self.num_variables, embedding_dim=self.embedding_dim,
                                      sequence_length=self.seq_length, batch_size=self.batch_size)

    def test_forward(self):
        # Test the forward() method by passing a random tensor of the expected shape
        x = rand(self.batch_size, self.num_variables, self.embedding_dim)
        output = forward(x)
        expected_shape = (self.batch_size, self.num_variables * self.embedding_dim, self.seq_length)
        self.assertEqual(output.shape, expected_shape)

    def test_positional_encoding_visualization(self):
        # Test the positional encoding visualization by checking that the plot is not empty
        from matplotlib import pyplot as plt
        import os
        x = rand(self.batch_size, self.num_variables, self.embedding_dim)
        x = self.pe(x)
        plt.figure(figsize=(16, 6))
        plt.plot(x[0].t().detach().numpy())
        plt.title('Positional Encoding')
        plt.xlabel('Position')
        plt.ylabel('Value')
        folder = 'test_positional_embedding'
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.savefig(f'{folder}/visualization.png')
        self.assertNotEqual(len(plt.gcf().axes), 0)


if __name__ == '__main__':
    unittest.main()
