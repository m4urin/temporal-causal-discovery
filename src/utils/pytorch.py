import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR, ConstantLR
from torch.autograd import Variable
from tqdm import trange

from definitions import DEVICE

warnings.filterwarnings("ignore", category=UserWarning, message=r'.*scheduler\.step\(\).*')


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Counts the number of parameters in a PyTorch model.

    Args:
        model: A PyTorch model to count the parameters of.
        trainable_only: A boolean flag indicating whether to count only trainable parameters.
            If True, only parameters with requires_grad=True will be counted.

    Returns:
        An integer representing the total number of parameters in the model.
    """
    if trainable_only:
        # count only trainable parameters
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # count all parameters, including non-trainable parameters
        return sum(p.numel() for p in model.parameters())


def generate_sine_wave(num_timesteps: int, amplitude: float = 1.0, frequency: float = 1.0) -> torch.Tensor:
    """
    Generate a time series representing one cycle of a sine wave.

    Args:
        num_timesteps (int): The number of timesteps in the time series.
        amplitude (float, optional): The amplitude of the sine wave. Defaults to 1.0.
        frequency (float, optional): The frequency of the sine wave. Defaults to 1.0.

    Returns:
        torch.Tensor: The generated time series.
    """
    timesteps = torch.arange(num_timesteps, dtype=torch.float32)

    angular_frequency = 2.0 * math.pi * frequency

    values = amplitude * torch.sin(angular_frequency * timesteps / num_timesteps)

    return values


def generate_random_sine_waves(num_samples: int, num_timesteps: int,
                               amplitude: tuple[float, float], frequency: tuple[float, float]) -> torch.Tensor:

    timesteps = torch.arange(num_timesteps, dtype=torch.float32).unsqueeze(0).expand(num_samples, -1)
    timesteps = timesteps + num_timesteps * torch.rand(num_samples, 1)

    random_frequencies = (frequency[1] - frequency[0]) * torch.rand(num_samples, 1) + frequency[0]
    angular_frequencies = 2.0 * math.pi * random_frequencies

    random_amplitudes = (amplitude[1] - amplitude[0]) * torch.rand(num_samples, 1) + amplitude[0]
    random_amplitudes *= 0.5
    values = random_amplitudes * torch.sin(angular_frequencies * timesteps / num_timesteps) + random_amplitudes

    return values


def interpolate_array(array, n, inclusive=False):
    """
    Interpolates an input array by generating additional values between each pair of adjacent elements.

    Args:
        array (list or numpy.ndarray): The input array.
        n (int): The number of values to generate between each pair of adjacent elements.

    Returns:
        numpy.ndarray: The interpolated array.

    Raises:
        ValueError: If the input array is empty or has only one element.

    Example:
        >>> interpolate_array([1, 2, 3], 2)
        array([1. , 1.5, 2. , 2.5])
    """
    array = np.array(array)

    if len(array) < 2:
        raise ValueError("Input array must have at least two elements.")

    # Calculate the total number of interpolated values needed
    num_interpolated_values = n * (len(array) - 1)

    if inclusive:
        num_interpolated_values += 1

    # Create an array to hold the interpolated values
    result = np.zeros(num_interpolated_values)

    # Interpolate between each pair of adjacent elements
    for i in range(len(array) - 1):
        start_value = array[i]
        end_value = array[i + 1]
        interpolated_values = np.linspace(start_value, end_value, n, endpoint=False)
        result[i * n: (i + 1) * n] = interpolated_values

    if inclusive:
        result[-1] = array[-1]

    return result


def exponential_scheduler_with_warmup(optimizer, start_factor, end_factor, warmup_iters, exp_iters, total_iters):
    warmup = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters-1)
    constant = ConstantLR(optimizer, factor=1.0, total_iters=total_iters - warmup_iters - exp_iters)
    exponential = ExponentialLR(optimizer, gamma=end_factor ** (1/exp_iters))
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup, constant, exponential],
                             milestones=[warmup_iters-1, total_iters - exp_iters])
    return scheduler


def find_minimum(model, input_size, output_size, k=1000, lr=1e-2, epochs=400,
                 min_input_val=None, max_input_val=None, show_loss=False, pbar=None, find_maximum=False):
    assert len(input_size) > 0

    obj = -1 if find_maximum else 1

    clamped = min_input_val is not None or max_input_val is not None

    model.eval()  # switch model to evaluation mode

    # Track the best minimum and maximum
    best_minimum = torch.full((k, *output_size), torch.inf, device=DEVICE)

    losses = []

    # Initialize a random point in the input space
    x = Variable(torch.randn(k, *input_size).to(DEVICE), requires_grad=True)

    # Minimize the output with gradient descent
    optimizer = torch.optim.Adam([x], lr=lr)
    scheduler = exponential_scheduler_with_warmup(optimizer, start_factor=0.1, end_factor=0.01,
                                                  warmup_iters=epochs // 50,
                                                  exp_iters=epochs // 2,
                                                  total_iters=epochs)

    if pbar is None:
        pbar = trange(epochs, desc="Find min/max..")
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = x
        if clamped:
            z = z.clamp(min=min_input_val, max=max_input_val)
        y = obj * model(z)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        best_minimum = torch.minimum(best_minimum, y.detach().view(k, *output_size, -1).mean(dim=-1))
        losses.append(loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(desc=f"Find min/max.. Loss={round(loss.item(), 4)}")
        pbar.update()

    if show_loss:
        plt.clf()
        plt.plot(losses, label='Train loss')
        plt.legend()
        plt.show()

    return obj * best_minimum.min(dim=0, keepdim=True).values


def find_extrema(model, input_size, output_size, k=1000, lr=1e-2, epochs=800,
                 min_input_val=None, max_input_val=None, show_loss=False, pbar=None):
    min_v = find_minimum(model, input_size, output_size, k, lr, epochs // 2,
                         min_input_val, max_input_val, show_loss, pbar,
                         find_maximum=False)
    max_v = find_minimum(model, input_size, output_size, k, lr, epochs - (epochs // 2),
                         min_input_val, max_input_val, show_loss, pbar,
                         find_maximum=True)
    return min_v, max_v


def fit_regression(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs=2000, lr=1e-2,
                   weight_decay=1e-6, show_loss=False, pbar=None):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = exponential_scheduler_with_warmup(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=0.01,
        warmup_iters=epochs // 20,
        exp_iters=epochs // 2,
        total_iters=epochs)

    loss_fn = nn.MSELoss()
    losses = []

    if pbar is None:
        pbar = trange(epochs, desc='Training..')

    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

        # Print loss for tracking progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(desc=f"Training.. Loss={round(loss.item(), 4)}")
        pbar.update()

    if show_loss:
        plt.clf()
        plt.plot(losses, label='Train loss')
        plt.legend()
        plt.show()

    return min(losses)


def fit_regression_model(model: nn.Module, model_original: nn.Module, data_size, epochs=2000, lr=1e-2,
                         weight_decay=1e-6, show_loss=False, pbar=None):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = exponential_scheduler_with_warmup(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=0.01,
        warmup_iters=epochs // 20,
        exp_iters=epochs // 2,
        total_iters=epochs)

    loss_fn = nn.MSELoss()
    losses = []

    if pbar is None:
        pbar = trange(epochs, desc='Training..')

    for epoch in range(epochs):
        # Zero the gradients
        optimizer.zero_grad()
        x = torch.randn(*data_size, device=DEVICE)
        with torch.no_grad():
            y = model_original(x)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        losses.append(loss.item())

        # Print loss for tracking progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            pbar.set_description(desc=f"Training.. Loss={round(loss.item(), 4)}")
        pbar.update()

    if show_loss:
        plt.clf()
        plt.plot(losses, label='Train loss')
        plt.legend()
        plt.show()

    return min(losses)


if __name__ == '__main__':
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(10, 10)

    total_epochs = 500
    _model = Model()
    optim = SGD(params=_model.parameters(), lr=1.0)
    s = exponential_scheduler_with_warmup(optim, start_factor=0.5, end_factor=0.01,
                                          warmup_iters=total_epochs//20,
                                          exp_iters=total_epochs//2,
                                          total_iters=total_epochs)

    lrs = []
    for _ in range(total_epochs):
        lrs.append(optim.param_groups[0]['lr'])
        s.step()

    plt.plot(lrs)
    plt.show()
