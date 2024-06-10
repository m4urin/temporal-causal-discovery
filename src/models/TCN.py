import math
import torch
from torch import nn


class TCN(nn.Module):
    """
    This module defines a flexible Temporal Convolutional Network (TCN) class that can be configured with different
    variants of TCN models, including weight sharing and recurrent architectures.

    The TCN class allows for easy instantiation of TCN models based on the desired configuration. It supports the
    following TCN variants:

    - DefaultTCN: The basic TCN model without weight sharing or recurrence.
    - WeightSharingTCN: TCN model with weight sharing across the temporal layers.
    - RecurrentTCN: TCN model with recurrent temporal layers.
    - WeightSharingRecurrentTCN: TCN model with both weight sharing and recurrent temporal layers.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int, kernel_size: int = 2,
                 n_blocks: int = 2, n_layers: int = 2, groups: int = 1,
                 dropout: float = 0.0, weight_sharing: bool = False, recurrent: bool = False,
                 use_padding: bool = False, use_positional_embedding=False, **kwargs):
        """
        This module defines a flexible Temporal Convolutional Network (TCN) class that can be configured with different
        variants of TCN models, including weight sharing and recurrent architectures.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_dim (int): Hidden dimension of the model.
            kernel_size (int): Kernel size for the convolutional layers.
            n_blocks (int, optional): Number of blocks in the TCN (default: 2).
            n_layers (int, optional): Number of layers per block in the TCN (default: 2).
            groups (int, optional): Number of groups for each Conv1d layer (default: 1).
            dropout (float, optional): Dropout probability for the convolutional layers (default: 0.0).
            weight_sharing (bool, optional): Whether to use weight sharing in the TCN (default: False).
            recurrent (bool, optional): Whether to use recurrent temporal layers (default: False).
            use_padding (bool, optional): Whether to use padding in convolutional layers to get the same
                                          input and output size (default: False).
            use_positional_embedding (bool, optional): Whether to use a positional embedding in convolutional layers
                                                   for temporal awareness in contemporaneous systems (default: False).
        """
        super().__init__()

        # Prepare a dictionary to collect the arguments
        config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'hidden_dim': hidden_dim,
            'kernel_size': kernel_size,
            'n_blocks': n_blocks,
            'n_layers': n_layers,
            'groups': groups,
            'dropout': dropout,
            'use_padding': use_padding,
            'use_positional_embedding': use_positional_embedding
        }

        # Create the correct variant of the TCN based on the flags
        if weight_sharing and recurrent:
            self.tcn = WeightSharingRecurrentTCN(**config)
        elif weight_sharing:
            self.tcn = WeightSharingTCN(**config)
        elif recurrent:
            self.tcn = RecurrentTCN(**config)
        else:
            self.tcn = DefaultTCN(**config)

        self.receptive_field = (2 ** n_blocks - 1) * n_layers * (kernel_size - 1) + 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the TCN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after passing through the TCN model.
        """
        return self.tcn(self.dropout(x))


class DefaultTCN(nn.Module):
    """
    A Temporal Convolutional Network (TCN) model that consists of a stack of Temporal Blocks.
    The TCN is designed to take in temporal sequences of data and produce a prediction for each time step.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int,
                 kernel_size: int, n_blocks: int = 2, n_layers: int = 1,
                 groups: int = 1, dropout: float = 0.0, use_padding: bool = False,
                 use_positional_embedding: bool = False):
        super().__init__()
        assert hidden_dim % groups == 0, "'hidden_dim' should be a multiple of 'groups'"

        # Calculate the total number of layers in the TCN
        self.n_total_layers = n_blocks * n_layers + 1

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
                    n_layers=n_layers,
                    dropout=dropout,
                    use_residual=i > 0,
                    use_padding=use_padding,
                    use_positional_embedding=i == 0 and use_positional_embedding)
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

        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):
        return self.network(x)


class RecurrentTCN(nn.Module):
    """
    A Recurrent Temporal Convolutional Network (Rec-TCN) model that consists of a first Temporal Block
    followed by a series of Recurrent Blocks. The Rec-TCN is designed to take in temporal sequences of data
    and produce a prediction for each time step.

    This version of the Rec-TCN has much fewer number of parameters compared to a general TCN when the number
    of layers is large.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int,
                 kernel_size: int, n_blocks: int = 3, n_layers: int = 1,
                 groups: int = 1, dropout: float = 0.0, use_padding: bool = False,
                 use_positional_embedding: bool = False):
        super().__init__()

        self.n_blocks = n_blocks

        assert self.n_blocks >= 3, 'if n_blocks is smaller than 3, please use a regular TCN'

        # Define the first TCN block
        self.first_block = TemporalBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers,
            dropout=dropout,
            use_residual=False,
            use_padding=use_padding,
            use_positional_embedding=use_positional_embedding
        )

        # Define the recurrent block
        self.recurrent = TemporalBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=1,
            groups=groups,
            n_layers=n_layers,
            dropout=dropout,
            use_padding=use_padding
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


class WeightSharingTCN(nn.Module):
    """
    A Weight Sharing Temporal Convolutional Network (WS-TCN) implementation. Weight sharing reduces
    the number of parameters and makes the model more efficient. Positional encoding provides information
    about the position of each element in the input sequence. Recurrent option can be used for even less parameters.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int,
                 kernel_size: int, n_blocks: int = 3, n_layers: int = 1,
                 groups: int = 1, dropout: float = 0.0, use_padding: bool = False,
                 use_positional_embedding: bool = False):
        super().__init__()

        assert hidden_dim % groups == 0, "The hidden dimension must be divisible by the number of groups"
        assert n_blocks >= 2, 'if n_blocks is smaller than 2, please use a regular TCN'

        self.groups = groups
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
            n_layers=n_layers,
            dropout=dropout,
            use_residual=False,
            use_padding=use_padding,
            use_positional_embedding=use_positional_embedding
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
                    n_layers=n_layers,
                    dropout=dropout,
                    use_padding=use_padding)
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

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        x = self.first_block(x)  # (batch_size, hidden_dim, seq_len)

        # Reshape input tensor to match the expected shape of the TCN
        # (batch_size * groups, hidden_dim // groups, seq_len)
        x = x.reshape(-1, self.hidden_dim // self.groups, x.size(-1))

        # Apply the TCN to the input tensor
        # (batch_size * groups, out_channels // groups, seq_len)
        x = self.weight_shared(x)

        # Reshape the output tensor to match the expected shape of the WS-TCN
        x = x.reshape(batch_size, -1, x.size(-1))  # (batch_size, out_channels, seq_len)

        # make the prediction
        x = self.prediction(x)

        return x


class WeightSharingRecurrentTCN(nn.Module):
    """
    A Recurrent Temporal Convolutional Network (Rec-TCN) model that consists of a first Temporal Block
    followed by a series of Recurrent Blocks. The Rec-TCN is designed to take in temporal sequences of data
    and produce a prediction for each time step.

    This version of the Rec-TCN has much fewer number of parameters compared to a general TCN when the number
    of layers is large.
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_dim: int,
                 kernel_size: int, n_blocks: int = 3, n_layers: int = 1,
                 groups: int = 1, dropout: float = 0.0, use_padding: bool = False,
                 use_positional_embedding: bool = False):
        super().__init__()
        assert n_blocks >= 3, 'if n_blocks is smaller than 3, please use a regular WS-TCN'
        self.groups = groups
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
            n_layers=n_layers,
            dropout=dropout,
            use_residual=False,
            use_padding=use_padding,
            use_positional_embedding=use_positional_embedding
        )

        # Define the recurrent block
        self.recurrent_weight_shared = TemporalBlock(
            in_channels=hidden_dim // groups,
            out_channels=hidden_dim // groups,
            kernel_size=kernel_size,
            dilation=1,
            groups=1,
            n_layers=n_layers,
            dropout=dropout,
            use_padding=use_padding
        )

        # Create the final prediction layer
        self.predictions = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Pass input tensor through the first TCN block
        x = self.first_block(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Reshape input tensor to match the expected shape of the TCN
        # (batch_size * groups, hidden_dim // groups, seq_len)
        x = x.reshape(-1, self.hidden_dim // self.groups, x.size(-1))

        # Pass input tensor through the residual blocks
        for i in range(self.n_blocks - 1):
            # Update the dilation factor for the Recurrent Block
            self.recurrent_weight_shared.dilation = 2 ** (i + 1)

            # Pass input tensor through the Recurrent Block
            x = self.recurrent_weight_shared(x)
            x = self.relu(x)
            x = self.dropout(x)

        # Reshape the output tensor to match the expected shape of the WS-TCN
        x = x.reshape(batch_size, -1, x.size(-1))  # (batch_size, out_channels, seq_len)

        # Pass the output through the final prediction layer
        return self.predictions(x)


class TemporalBlock(nn.Module):
    """
    A PyTorch module that implements a Temporal Block.

    This module consists of a sequence of 1D convolutional layers with dilations, followed by ReLU activations
    and dropout layers. The number of layers, kernel size, dilation, and dropout rate can be configured
    during initialization. Additionally, it allows external configuration of the 'dilation' attribute,
    facilitating its use in a recurrent context (necessary for RecurrentTCN).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        dilation (int, optional): Dilation factor for the convolutional layers (default: 1).
        groups (int, optional): Number of groups to split the input and output channels into (default: 1).
        n_layers (int, optional): Number of convolutional layers in the block (default: 1).
        dropout (float, optional): Dropout rate to use between layers (default: 0.0).
        use_padding (bool, optional): Whether to use padding in convolutional layers (default: False).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int = 1, groups: int = 1, n_layers: int = 1, dropout: float = 0.0,
                 use_residual=True, use_padding: bool = False, use_positional_embedding: bool = False):
        super().__init__()

        assert n_layers > 0, "Number of layers should be 1 or greater."

        # Set class variables
        self.n_layers = n_layers
        self.dilation = dilation
        self.kernel_size = kernel_size

        # Define dropout, and activation functions
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Define the network architecture
        self.convolutions = nn.ModuleList()
        # Create down-sample branch if input and output channels differ
        self.down_sample = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=groups
        ) if in_channels != out_channels and use_residual else None

        self.use_residual = use_residual
        self.use_padding = use_padding
        self.positional_embedding = None
        if use_positional_embedding:
            self.positional_embedding = PositionalEmbedding(out_channels, groups=groups)

        for i in range(n_layers):
            self.convolutions.append(nn.utils.parametrizations.weight_norm(nn.Conv1d(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups
            )))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the Temporal Block to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, new_sequence_length).
        """
        # Set correct dilations
        for conv in self.convolutions:
            conv.dilation = self.dilation

        # Save input tensor to be used as identity
        identity = x

        # If necessary, pass input tensor through down-sample branch
        if self.down_sample is not None:
            identity = self.down_sample(identity)

        for i in range(self.n_layers - 1):
            # Apply the layer
            if self.use_padding:
                x = nn.functional.pad(x, (self.dilation * (self.kernel_size - 1), 0), 'constant', 0)
            x = self.convolutions[i](x)
            x = self.relu(x)
            if i == 0 and self.positional_embedding is not None:
                x = self.positional_embedding(x)
            x = self.dropout(x)

        # Do not apply relu/dropout to last layer
        if self.use_padding:
            x = nn.functional.pad(x, (self.dilation * (self.kernel_size - 1), 0), 'constant', 0)
        x = self.convolutions[-1](x)

        # Add residual connection to the data
        if self.use_residual:
            x = x + identity[..., -x.size(-1):]

        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim, groups=1, max_length=3000):
        super().__init__()
        h = hidden_dim // groups
        # Create the positional embedding
        pos_embedding = torch.zeros(1, h, max_length)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, h, 2).float() * (-math.log(10000.0) / h))
        pos_embedding[0, 0::2, :] = torch.sin(position * div_term).t()
        pos_embedding[0, 1::2, :] = torch.cos(position * div_term).t()
        pos_embedding *= 0.5
        pos_embedding = pos_embedding.repeat(1, groups, 1)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding[..., :x.size(-1)]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    plt.plot(PositionalEmbedding(3 * 8, 3, max_length=3000).pos_embedding[0].t())
    plt.show()
    model = TCN(in_channels=3, out_channels=3*3, hidden_dim=3*8, groups=3,
                use_padding=True, use_positional_embedding=True)
    print(model(torch.randn(1, 3, 200)).shape)
