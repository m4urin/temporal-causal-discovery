import torch

from src.models.temporal_causal_model import TemporalCausalModel


def monte_carlo_dropout(model: TemporalCausalModel, x: torch.Tensor, samples: int = 300, max_batch_size: int = 128):
    """
    x is of size (batch_size, num_variables, sequence_length)

    It performs Monte Carlo dropout to estimate the uncertainty of the model's predictions by sampling the output of the model with dropout enabled. The function takes in a PyTorch tensor x and an integer samples, which represents the number of Monte Carlo samples to take. It then temporarily enables dropout by setting self.training = True, samples the output of the model with dropout enabled samples times, computes the mean and standard deviation of the model's output for each tensor attribute using PyTorch's mean and std functions, and returns two instances of ModelOutput containing the mean and standard deviation of the model's output.
    The mean_data and std_data dictionaries are created by copying the attributes of the first element of the model_outputs list, which is a ModelOutput object, using the __dict__() method. These dictionaries are then modified by replacing the tensor attributes with their mean and standard deviation, respectively.
    After the computation is finished, the self.training mode is restored to its original value by setting it to mode.
    """
    assert samples > 0, 'at least one sample to compute'
    assert max_batch_size > 0, "TODO"
    mode = model.training
    model.train()  # enable dropout

    batch_size, num_variables, sequence_length = x.shape

    with torch.no_grad():
        # size (samples * batch_size, ...)
        x = x.unsqueeze(0).expand(samples, -1, -1, -1).view(-1, num_variables, sequence_length)
        model_output = model.forward(x, max_batch_size=max_batch_size)
        for k, v in model_output.items():
            setattr(model_output, k, v.view(samples, batch_size, *v.shape[1:]))
        return model_output

"""
mean_data, std_data = {}, {}
for k, v in model_output.items():
    v = v.view(samples, batch_size, *v.shape[1:])
    mean_data[k] = v.mean(dim=0)
    std_data[k] = v.std(dim=0)

model_output_mean = type(model_output)(**mean_data)
model_output_std = type(model_output)(**std_data)

model.train(mode=mode)  # restore mode
return model_output_mean, model_output_std
"""
