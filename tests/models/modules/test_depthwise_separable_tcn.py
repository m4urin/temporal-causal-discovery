import unittest

from src.models.modules.depthwise_separable_tcn import *


class TestDepthwiseSeparableTCN(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.groups = 3
        self.in_channels = 3
        self.out_channels = 48
        self.sequence_length = 100
        self.kernel_size = 3
        self.n_layers = 2
        self.block_layers = 2
        self.dropout = 0.0

        self.tcn = DepthwiseSeparableTCN(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel_size,
                                         groups=self.groups,
                                         n_blocks=self.n_layers,
                                         n_layers_per_block=self.block_layers,
                                         dropout=self.dropout)

    def test_output_shape(self):
        x = torch.randn(self.batch_size, self.in_channels, self.sequence_length)
        out = self.tcn(x)
        self.assertEqual(out.shape, (self.batch_size, self.out_channels, self.sequence_length))

    def test_receptive_field(self):
        self.assertEqual(self.tcn.receptive_field, 13)

    def test_total_layers(self):
        self.assertEqual(self.tcn.n_total_layers, 4)

    def test_num_channels(self):
        for i, layer in enumerate(self.tcn.network):
            in_ch = self.in_channels if i == 0 else self.out_channels
            self.assertEqual(layer.block[0].in_channels, in_ch)
            self.assertEqual(layer.block[0].out_channels, self.out_channels)


if __name__ == '__main__':
    unittest.main()
