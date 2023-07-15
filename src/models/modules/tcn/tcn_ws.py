import torch

from src.models.modules.temporal.positional_embedding import PositionalEmbedding
from src.models.modules.tcn.base_tcn import BaseTCN
from src.models.modules.tcn.tcn_rec import RecTCN
from src.models.modules.temporal.temporal_module import TemporalModule


class WSTCN(TemporalModule):
    """
    A Weight Sharing Temporal Convolutional Network (WS-TCN) implementation. Weight sharing reduces
    the number of parameters and makes the model more efficient. Positional encoding provides information
    about the position of each element in the input sequence. Recurrent option can be used for even less parameters.

    Args:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - hidden_dim (int): Hidden dimension of the model.
    - kernel_size (int): Kernel size for the convolutional layers.
    - max_sequence_length (int): Maximum length of the input sequence.
    - n_blocks (int): Number of blocks in the TCN.
    - n_layers_per_block (int): Number of layers per block in the TCN.
    - groups (int): Number of groups to use for weight sharing.
    - dropout (float): Dropout probability for the convolutional layers.
    - use_recurrent (bool): Whether to use a recurrent TCN.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        kernel_size: int,
        max_sequence_length: int,
        n_blocks: int = 3,
        n_layers_per_block: int = 1,
        groups: int = 1,
        dropout: float = 0.0,
        use_recurrent: bool = False
    ):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1 )

        assert hidden_dim % groups == 0, "The hidden dimension must be divisible by the number of groups"
        self.hidden_dim = hidden_dim

        # up sampling and adding an embedding
        self.positional_embedding = PositionalEmbedding(in_channels, hidden_dim, max_sequence_length, groups)

        # Initialize tcn
        tcn_constructor = RecTCN if use_recurrent else BaseTCN
        self.tcn: TemporalModule = tcn_constructor(
            hidden_dim // groups,
            out_channels // groups,
            hidden_dim // groups,
            kernel_size,
            n_blocks,
            n_layers_per_block,
            groups=1,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the WS-TCN.

        Args:
        - x (torch.Tensor): The input tensor with shape (batch_size, in_channels, seq_length).

        Returns:
        - torch.Tensor: The output tensor of the WS-TCN with shape (batch_size, out_channels, seq_length).
        """

        batch_size, _, seq_length = x.size()

        x = self.positional_embedding(x)  # (batch_size, hidden_dim, seq_len)

        # Reshape input tensor to match the expected shape of the TCN
        # (batch_size * groups, hidden_dim // groups, seq_len)
        x = x.view(-1, self.hidden_dim // self.groups, seq_length)

        # Apply the TCN to the input tensor
        # (batch_size * groups, out_channels // groups, seq_len)
        x = self.tcn(x)

        # Reshape the output tensor to match the expected shape of the WS-TCN
        x = x.view(batch_size, self.out_channels, seq_length)  # (batch_size, out_channels, seq_len)

        return x

    def get_embeddings(self):
        return self.positional_embedding.get_embeddings()


if __name__ == '__main__':
    from src.utils.pytorch import count_parameters

    tcn = WSTCN(3, 3, 96, 2, 1000, 3, 2, 3, 0.0, False)
    if isinstance(tcn, WSTCN):
        print(tcn.get_embeddings().size())

    print(tcn)
    print(count_parameters(tcn))
