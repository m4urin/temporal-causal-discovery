import torch

from src.models.model_outputs import ModelOutput


def monte_carlo_dropout(model: torch.nn.Module, x: torch.Tensor,
                        samples: int = 300, max_batch_size: int = 128) -> ModelOutput:
    """
    Monte Carlo dropout estimation for a TemporalCausalModel.

    Args:
        model (torch.nn.Module): The temporal causal model to use.
        x (torch.Tensor): The input tensor, of size (batch_size, num_variables, sequence_length).
        samples (int): The number of samples to take.
        max_batch_size (int): The maximum batch size to use for each forward pass.

    Returns:
        A dictionary of output tensors from the model, with keys corresponding to the names of each output.
    """
    # Ensure the number of samples is valid
    assert samples > 0, 'Must have at least one sample to compute'
    # Ensure the batch size is valid
    assert max_batch_size > 0, 'Max batch size must be positive'

    # Save the current mode of the model, and switch to training mode
    mode = model.training
    model.train()

    # Unpack the input shape
    batch_size, num_variables, sequence_length = x.shape

    # Expand the input tensor to take multiple samples
    expanded_inputs = x.unsqueeze(0).expand(samples, -1, -1, -1).view(-1, num_variables, sequence_length)
    # Run the forward pass on the expanded inputs
    with torch.no_grad():
        model_output = model.forward(expanded_inputs, max_batch_size=max_batch_size)

        # Reshape the model output tensors to separate samples and batch dimension
        for key, value in model_output.items():
            setattr(model_output, key, value.view(samples, batch_size, *value.shape[1:]))

    # Restore the model to its original mode
    model.train(mode=mode)

    return model_output
