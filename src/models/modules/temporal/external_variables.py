import torch

from src.models.modules.temporal.positional_embedding import PositionalEmbedding
from src.models.modules.temporal.temporal_module import TemporalModule


class ExternalVariables(TemporalModule):
    def __init__(self, num_channels: int, groups: int, num_ext: int, max_len: int):
        """
        Initializes the ExternalVariables module.

        Args:
            num_channels (int): Number of input channels.
            groups (int): Number of groups for channel splitting.
            num_ext (int): Number of external variables.
            max_len (int): Maximum length for positional embeddings.
        """
        super().__init__(num_channels, num_channels, groups)
        self.positional_embedding = PositionalEmbedding(0, num_ext * self.in_dim, max_len, groups=num_ext)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ExternalVariables module.

        Args:
            x (torch.Tensor): Tensor of size (batch_size, groups * (in_channels // groups), sequence_length).

        Returns:
            torch.Tensor: Tensor of size (batch_size, (groups + num_ext_variables) * (in_channels // groups),
                sequence_length).
        """
        # Compute positional embeddings
        emb = self.positional_embedding()

        # Concatenate input tensor with positional embeddings
        x = torch.cat((x, emb[..., :x.size(-1)].expand(x.size(0), -1, -1)), dim=1)

        return x

    def get_embeddings(self) -> torch.Tensor:
        """
        Retrieves the positional embeddings.

        Returns:
            torch.Tensor: Tensor of size (max_len, num_ext * in_dim).
        """
        return self.positional_embedding.get_embeddings()


if __name__ == '__main__':
    # Define input parameters
    batch_size, input_dim, n_groups, sequence_length, num_extra = 1, 32, 3, 100, 2

    # Create input tensor
    data = torch.zeros(batch_size, input_dim * n_groups, sequence_length)

    # Create ExternalVariables module instance
    module = ExternalVariables(input_dim * n_groups, n_groups, num_extra, sequence_length)

    print("Input size:", data.size())
    result = module(data)
    print("Output size:", result.size())

    # Reshape and print the result
    reshaped_result = result.view(batch_size, n_groups + num_extra, input_dim, sequence_length)
    print("Reshaped result:", reshaped_result.size())
