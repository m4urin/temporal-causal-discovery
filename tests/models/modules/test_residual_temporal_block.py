import unittest

from src.models.modules.residual_temporal_block import *


class TestResidualTemporalBlock(unittest.TestCase):
    def setUp(self):
        self.rtb = ResidualTemporalBlock(10, 20, 1,
                                nn.Conv1d(10, 20, kernel_size=3, padding=1),
                                nn.BatchNorm1d(20),
                                nn.ReLU(),
                                nn.Conv1d(20, 20, kernel_size=3, padding=1),
                                nn.BatchNorm1d(20),
                                nn.ReLU(),
                                nn.Conv1d(20, 20, kernel_size=3, padding=1),
                                nn.BatchNorm1d(20),
                                nn.ReLU())
        self.x = torch.randn(32, 10, 100)

    def test_output_shape(self):
        y = self.rtb(self.x)
        self.assertEqual(y.shape, (32, 20, 100))

    def test_skip_connection(self):
        identity = self.x
        y = self.rtb(self.x)
        for module in self.rtb.temporal_module:
            if isinstance(module, nn.Module):
                identity = module(identity)
        if self.rtb.down_sample is not None:
            identity = self.rtb.down_sample(self.x)
        print(identity[0, 0, :10])
        print(y[0, 0, :10])
        self.assertEqual(torch.allclose(y, identity))


if __name__ == '__main__':
    unittest.main()