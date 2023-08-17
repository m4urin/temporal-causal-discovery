import torch
from torch import nn

from src.models.modules.temporal.temporal_block import TemporalBlock
from src.models.modules.temporal.temporal_module import TemporalModule


class TCN(nn.Module):
    """
    This module defines a flexible Temporal Convolutional Network (TCN) class that can be configured with different
    variants of TCN models, including weight sharing and recurrent architectures.

    The TCN class allows for easy instantiation of TCN models based on the desired configuration. It supports the
    following TCN variants:
    - BaseTCN: The basic TCN model without weight sharing or recurrence.
    - WSTCN (Weight Sharing TCN): TCN model with weight sharing across the temporal layers.
    - RecTCN (Recurrent TCN): TCN model with recurrent temporal layers.
    - WSRecTCN (Weight Sharing Recurrent TCN): TCN model with both weight sharing and recurrent temporal layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_dim (int): Hidden dimension of the model.
        kernel_size (int): Kernel size for the convolutional layers.
        n_blocks (int, optional): Number of blocks in the TCN (default: 2).
        n_layers_per_block (int, optional): Number of layers per block in the TCN (default: 2).
        groups (int, optional): Number of groups for each Conv1d layer (default: 1).
        dropout (float, optional): Dropout probability for the convolutional layers (default: 0.0).
        weight_sharing (bool, optional): Whether to use weight sharing in the TCN (default: False).
        recurrent (bool, optional): Whether to use recurrent temporal layers (default: False).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_dim: int,
                 kernel_size: int,
                 n_blocks: int = 2,
                 n_layers_per_block: int = 2,
                 groups: int = 1,
                 dropout: float = 0.0,
                 weight_sharing: bool = False,
                 recurrent: bool = False):
        super().__init__()
        if weight_sharing and recurrent:
            self.tcn = WeightSharingRecurrentTCN(in_channels, out_channels, hidden_dim, kernel_size,
                                                 n_blocks, n_layers_per_block, groups, dropout)
        elif weight_sharing:
            self.tcn = WeightSharingTCN(in_channels, out_channels, hidden_dim, kernel_size,
                                        n_blocks, n_layers_per_block, groups, dropout)
        elif recurrent:
            self.tcn = RecurrentTCN(in_channels, out_channels, hidden_dim, kernel_size,
                                    n_blocks, n_layers_per_block, groups, dropout)
        else:
            self.tcn = DefaultTCN(in_channels, out_channels, hidden_dim, kernel_size,
                                  n_blocks, n_layers_per_block, groups, dropout)

    def forward(self, x):
        """
        Forward pass through the TCN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after passing through the TCN model.
        """
        return self.tcn(x)


class DefaultTCN(TemporalModule):
    """
    A Temporal Convolutional Network (TCN) model that consists of a stack of Temporal Blocks.
    The TCN is designed to take in temporal sequences of data and produce a prediction for each time step.

    Args:
        in_channels (int): The number of input channels for the first Temporal Block.
        out_channels (int): The number of output channels for the last Conv1d layer.
        hidden_dim (int): The number of hidden channels for each Temporal Block.
        kernel_size (int): The kernel size for each Temporal Block.
        n_blocks (int, optional): The number of Temporal Blocks in the TCN (default: 1).
        n_layers_per_block (int, optional): The number of layers in each Temporal Block (default: 1).
        groups (int, optional): The number of groups for each Conv1d layer (default: 1).
        dropout (float, optional): The dropout probability for each Temporal Block (default: 0.0).
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_dim: int,
            kernel_size: int,
            n_blocks: int = 2,
            n_layers_per_block: int = 1,
            groups: int = 1,
            dropout: float = 0.0
    ):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1)
        assert hidden_dim % groups == 0, "'hidden_dim' should be a multiple of 'groups'"

        # Calculate the total number of layers in the TCN
        self.n_total_layers = n_blocks * n_layers_per_block + 1

        # Initialize the activation and dropout layers
        relu = nn.ReLU()
        dropout_layer = nn.Dropout(dropout)

        # Create the stack of Temporal Blocks
        modules = []
        for i in range(n_blocks):
            modules.append(
                TemporalBlock(
                    in_channels=in_channels if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    groups=groups,
                    n_layers=n_layers_per_block,
                    dropout=dropout,
                    use_residual=i > 0)
            )
            modules.append(relu)
            modules.append(dropout_layer)

        # Create the final prediction layer
        modules.append(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups
            )
        )

        self.seq = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, seq_len).
        """
        return self.seq(x)


class RecurrentTCN(TemporalModule):
    """
    A Recurrent Temporal Convolutional Network (Rec-TCN) model that consists of a first Temporal Block
    followed by a series of Recurrent Blocks. The Rec-TCN is designed to take in temporal sequences of data
    and produce a prediction for each time step.

    This version of the Rec-TCN has much fewer number of parameters compared to a general TCN when the number
    of layers is large.

    Args:
        in_channels (int): The number of input channels for the first Temporal Block.
        out_channels (int): The number of output channels for the last Conv1d layer.
        hidden_dim (int): The number of hidden channels for each Temporal Block.
        kernel_size (int): The kernel size for each Temporal Block.
        n_blocks (int, optional): The number of Recurrent Blocks in the Rec-TCN (default: 3).
        n_layers_per_block (int, optional): The number of layers in each Temporal Block (default: 1).
        groups (int, optional): The number of groups for each Conv1d layer (default: 1).
        dropout (float, optional): The dropout probability for each Temporal Block (default: 0.0).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        kernel_size: int,
        n_blocks: int = 3,
        n_layers_per_block: int = 1,
        groups: int = 1,
        dropout: float = 0.0
    ):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1)

        self.n_blocks = n_blocks

        assert self.n_blocks >= 3, 'if n_blocks is smaller than 3, please use a regular TCN'

        # Define the first TCN block
        self.first_block = TemporalBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers_per_block,
            dropout=dropout,
            use_residual=False
        )

        # Define the recurrent block
        self.recurrent = TemporalBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers_per_block,
            dropout=dropout
        )

        # Define the dropout and ReLU layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Create the final prediction layer
        self.predictions = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups
        )

    def forward(self, x):
        """
        Forward pass through the Rec-TCN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, seq_len).
        """
        # Pass input tensor through the first TCN block
        x = self.first_block(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass input tensor through the residual blocks
        for i in range(self.n_blocks - 1):
            # Update the dilation factor for the Recurrent Block
            self.recurrent.dilation = 2 ** (i + 1)

            # Pass input tensor through the Recurrent Block
            x = self.recurrent(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Pass the output through the final prediction layer
        return self.predictions(x)


class WeightSharingTCN(TemporalModule):
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
        n_blocks: int = 3,
        n_layers_per_block: int = 1,
        groups: int = 1,
        dropout: float = 0.0
    ):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1 )

        assert hidden_dim % groups == 0, "The hidden dimension must be divisible by the number of groups"
        assert self.n_blocks >= 2, 'if n_blocks is smaller than 2, please use a regular TCN'

        self.hidden_dim = hidden_dim

        # Initialize the activation and dropout layers
        relu = nn.ReLU()
        dropout_layer = nn.Dropout(dropout)

        # Define the first TCN block
        self.first_block = nn.Sequential(TemporalBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers_per_block,
            dropout=dropout,
            use_residual=False
        ), relu, dropout_layer)

        # Create the stack of Temporal Blocks
        modules = []
        for i in range(1, n_blocks):
            modules.append(
                TemporalBlock(
                    in_channels=hidden_dim // groups,
                    out_channels=hidden_dim // groups,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    groups=1,
                    n_layers=n_layers_per_block,
                    dropout=dropout)
            )
            modules.append(relu)
            modules.append(dropout_layer)

        self.weight_shared = nn.Sequential(*modules)

        # Create the final prediction layer
        self.prediction = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                groups=groups
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

        x = self.first_block(x)  # (batch_size, hidden_dim, seq_len)

        # Reshape input tensor to match the expected shape of the TCN
        # (batch_size * groups, hidden_dim // groups, seq_len)
        x = x.reshape(-1, self.hidden_dim // self.groups, seq_length)

        # Apply the TCN to the input tensor
        # (batch_size * groups, out_channels // groups, seq_len)
        x = self.weight_shared(x)

        # Reshape the output tensor to match the expected shape of the WS-TCN
        x = x.reshape(batch_size, self.out_channels, seq_length)  # (batch_size, out_channels, seq_len)

        # make the prediction
        x = self.prediction(x)

        return x


class WeightSharingRecurrentTCN(TemporalModule):
    """
    A Recurrent Temporal Convolutional Network (Rec-TCN) model that consists of a first Temporal Block
    followed by a series of Recurrent Blocks. The Rec-TCN is designed to take in temporal sequences of data
    and produce a prediction for each time step.

    This version of the Rec-TCN has much fewer number of parameters compared to a general TCN when the number
    of layers is large.

    Args:
        in_channels (int): The number of input channels for the first Temporal Block.
        out_channels (int): The number of output channels for the last Conv1d layer.
        hidden_dim (int): The number of hidden channels for each Temporal Block.
        kernel_size (int): The kernel size for each Temporal Block.
        n_blocks (int, optional): The number of Recurrent Blocks in the Rec-TCN (default: 3).
        n_layers_per_block (int, optional): The number of layers in each Temporal Block (default: 1).
        groups (int, optional): The number of groups for each Conv1d layer (default: 1).
        dropout (float, optional): The dropout probability for each Temporal Block (default: 0.0).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        kernel_size: int,
        n_blocks: int = 3,
        n_layers_per_block: int = 1,
        groups: int = 1,
        dropout: float = 0.0
    ):
        super().__init__(in_channels, out_channels, groups,
                         receptive_field=(2 ** n_blocks - 1) * n_layers_per_block * (kernel_size - 1) + 1)
        assert n_blocks >= 3, 'if n_blocks is smaller than 3, please use a regular WS-TCN'

        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim

        # Define the dropout and ReLU layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Define the first TCN block
        self.first_block = TemporalBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers_per_block,
            dropout=dropout,
            use_residual=False
        )

        # Define the recurrent block
        self.recurrent_weight_shared = TemporalBlock(
            in_channels=hidden_dim // groups,
            out_channels=hidden_dim // groups,
            kernel_size=kernel_size,
            dilation=1,
            groups=1,
            n_layers=n_layers_per_block,
            dropout=dropout
        )

        # Create the final prediction layer
        self.predictions = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups
        )

    def forward(self, x):
        """
        Forward pass through the Rec-TCN model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, seq_len).
        """
        batch_size, _, seq_length = x.size()

        # Pass input tensor through the first TCN block
        x = self.first_block(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Reshape input tensor to match the expected shape of the TCN
        # (batch_size * groups, hidden_dim // groups, seq_len)
        x = x.reshape(-1, self.hidden_dim // self.groups, seq_length)

        # Pass input tensor through the residual blocks
        for i in range(self.n_blocks - 1):
            # Update the dilation factor for the Recurrent Block
            self.recurrent.dilation = 2 ** (i + 1)

            # Pass input tensor through the Recurrent Block
            x = self.recurrent(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Reshape the output tensor to match the expected shape of the WS-TCN
        x = x.reshape(batch_size, -1, seq_length)  # (batch_size, out_channels, seq_len)

        # Pass the output through the final prediction layer
        return self.predictions(x)
