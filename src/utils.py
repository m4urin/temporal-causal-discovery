import argparse
import json
import math
import os
import pickle
import bz2
import warnings

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.optim import lr_scheduler
from pathlib import Path
from collections import defaultdict
from hyperopt import hp
import time


# --------- TCN ---------

def generate_architecture_options(max_lags, marge, minimum_num_options=1,
                                  n_blocks=None, n_layers_per_block=None, kernel_size=None):
    """Generate architecture options based on given parameters."""
    def _receptive_field(b, n, k):
        return n * ((2 ** b) - 1) * (k - 1) + 1

    assert max_lags >= 2
    if n_blocks and n_layers_per_block and kernel_size:
        assert n_blocks >= 1 and n_layers_per_block >= 1 and kernel_size >= 2
        valids = [(n_blocks, n_layers_per_block, kernel_size)]
    else:
        blocks_range = (1, 100) if n_blocks is None else (n_blocks, n_blocks + 1)
        layers_range = (1, 2 + 1) if n_layers_per_block is None else (n_layers_per_block, n_layers_per_block + 1)
        kernel_range = (2, 1000) if kernel_size is None else (kernel_size, kernel_size + 1)

        layers_min = layers_range[0]
        kernel_min = kernel_range[0]

        result = set()
        while len(result) < minimum_num_options:
            for blocks in range(*blocks_range):
                if max_lags + marge > _receptive_field(blocks, layers_min, kernel_min) >= max_lags:
                    result.add((blocks, layers_min, kernel_min))
                    break
                for layers in range(*layers_range):
                    if max_lags + marge > _receptive_field(blocks, layers, kernel_min) >= max_lags:
                        result.add((blocks, layers, kernel_min))
                        break
                    for k_size in range(*kernel_range):
                        if max_lags + marge > _receptive_field(blocks, layers, k_size) >= max_lags:
                            result.add((blocks, layers, k_size))
                            break
            if marge > 3 * max_lags:
                break
            marge += 1

        valids = list(result)

    return [{'n_blocks': b, 'n_layers_per_block': l, 'kernel_size': k} for b, l, k in valids]


# --------- PyTorch Functions ---------

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


def weighted_mean(x: torch.Tensor, weights: torch.Tensor, dim, keepdim=False):
    return (weights * x).sum(dim=dim, keepdim=keepdim) / weights.sum(dim=dim, keepdim=keepdim)


def weighted_std(x: torch.Tensor, weights: torch.Tensor, dim, keepdim=False):
    total_weight = weights.sum(dim=dim, keepdim=True)
    weighted_mean = (weights * x).sum(dim=dim, keepdim=True) / total_weight
    if not keepdim:
        total_weight = total_weight.squeeze(dim)
    weighted_variance = (weights * (x - weighted_mean).pow(2)).sum(dim=dim, keepdims=keepdim) / total_weight
    return torch.sqrt(weighted_variance)


def pad(x: torch.Tensor, padding: tuple, dim: int = -1) -> torch.Tensor:
    """
    Pads a given tensor along a specified dimension with the provided padding values.

    Args:
        x (torch.Tensor): The input tensor to be padded.
        padding (tuple): A tuple containing the padding values for the left and right sides.
        dim (int, optional): The dimension along which padding should be applied. Default is -1.

    Returns:
        torch.Tensor: Padded tensor.
    """
    num_dims = x.dim()
    pad_sizes = [0] * 2 * num_dims
    index = 2 * ((-dim - 1) % num_dims)
    pad_sizes[index] = padding[0]  # Padding to the left
    pad_sizes[index + 1] = padding[1]  # Padding to the right
    return torch.nn.functional.pad(x, pad_sizes)


def sliding_window_std(x: torch.Tensor, window: tuple, dim: int = -1, correction: int = 1) -> torch.Tensor:
    """
    Computes the sliding window standard deviation of a tensor along a specified dimension.

    Parameters:
        x (torch.Tensor): Input tensor.
        window (tuple of int): The sliding window (left-inclusive, right). The sum of the tuple is the total window size.
        dim (int, optional): The dimension along which to compute the sliding window standard deviation. Default is the last dimension.
        correction (int, optional): A correction factor for the biased variance. Default is 1.

    Returns:
        torch.Tensor: Tensor of sliding window standard deviations.
    """
    tensor_size = x.size(dim)
    window_size = sum(window)

    x = pad(x, window, dim=dim)  # Assumes pad function is imported and defined elsewhere

    # Cumulative sum of x and x^2
    x_cumsum = torch.cumsum(x, dim=dim)
    x2_cumsum = torch.cumsum(x ** 2, dim=dim)

    # Sum of x and x^2 over the window
    x_sum = x_cumsum.narrow(dim, window_size, tensor_size) - x_cumsum.narrow(dim, 0, tensor_size)
    x2_sum = x2_cumsum.narrow(dim, window_size, tensor_size) - x2_cumsum.narrow(dim, 0, tensor_size)

    # Mean of x and x^2 over the window
    x_mean = x_sum.div(window_size)
    x2_mean = x2_sum.div(window_size)

    # Variance and standard deviation
    variance = x2_mean - x_mean ** 2

    if correction == 1:
        variance = window_size * variance / (window_size - correction)

    std_dev = torch.sqrt(variance)

    return std_dev


def weighted_sliding_window_std(x: torch.Tensor, weights: torch.Tensor,
                                window: tuple, dim: int = -1, correction: int = 1) -> torch.Tensor:
    """
    Computes the sliding window weighted standard deviation of a tensor along a specified dimension.

    Parameters:
        x (torch.Tensor): Input tensor.
        weights (torch.Tensor): Weights tensor with the same shape as x.
        window (tuple of int): The sliding window (left-inclusive, right). The sum of the tuple is the total window size.
        dim (int, optional): The dimension along which to compute the sliding window standard deviation. Default is the last dimension.
        correction (int, optional): A correction factor for the biased variance. Default is 1.

    Returns:
        torch.Tensor: Tensor of sliding window weighted standard deviations.
    """
    assert x.shape == weights.shape, "x and weights must have the same shape"

    tensor_size = x.size(dim)
    window_size = sum(window)

    x = pad(x, window, dim=dim)  # Assumes pad function is imported and defined elsewhere
    weights = pad(weights, window, dim=dim)

    # Weighted cumulative sum of x, x^2, and weights
    w_cumsum = torch.cumsum(weights, dim=dim)
    xw_cumsum = torch.cumsum(x * weights, dim=dim)
    x2w_cumsum = torch.cumsum((x ** 2) * weights, dim=dim)

    # Sum of weighted x, x^2, and weights over the window
    w_sum = w_cumsum.narrow(dim, window_size, tensor_size) - w_cumsum.narrow(dim, 0, tensor_size)
    xw_sum = xw_cumsum.narrow(dim, window_size, tensor_size) - xw_cumsum.narrow(dim, 0, tensor_size)
    x2w_sum = x2w_cumsum.narrow(dim, window_size, tensor_size) - x2w_cumsum.narrow(dim, 0, tensor_size)

    # Weighted mean of x, x^2, and weights over the window
    xw_mean = xw_sum.div(w_sum)
    x2w_mean = x2w_sum.div(w_sum)

    # Weighted variance and standard deviation
    variance = x2w_mean - xw_mean ** 2

    if correction == 1:
        w2_cumsum = torch.cumsum(weights.pow(2), dim=dim)
        w2_sum = w2_cumsum.narrow(dim, window_size, tensor_size) - w2_cumsum.narrow(dim, 0, tensor_size)
        variance = variance * w_sum / ((w_sum ** 2) - w2_sum)

    std_dev = torch.sqrt(variance)

    return std_dev


def entropy(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Computes the entropy of a probability distribution along a specified dimension of the input tensor.

    Args:
        x (torch.Tensor): The input tensor representing a probability distribution.
                          Each value should be greater or equal to 0.
        dim (int, optional): The dimension along which to compute the entropy.
            Defaults to -1.
        keepdim (bool, optional): Whether to keep the dimension of the output tensor.
            Defaults to False.

    Returns:
        torch.Tensor: The entropy tensor.

    Raises:
        IndexError: If the specified dimension is out of range.
    """

    # Compute the negative entropy by multiplying the input tensor with its logarithm
    # and dividing it by the logarithm of the tensor size along the specified dimension
    # Adds a small constant to the input tensor to avoid taking the logarithm of zero
    entropy_tensor = -x * torch.log(x + 1e-30) / math.log(x.size(dim))

    # Sum the entropy values along the specified dimension
    return entropy_tensor.sum(dim=dim, keepdim=keepdim)


def exponential_scheduler_with_warmup(total_iters, optimizer, start_factor=1.0, end_factor=1.0,
                                      warmup_ratio=0.0, cooldown_ratio=0.0):
    """
    Create a learning rate scheduler with warmup, constant, and exponential decay phases.

    Args:
        total_iters (int): The total number of iterations for the scheduler.
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
        start_factor (float): The initial learning rate factor.
        end_factor (float): The final learning rate factor.
        warmup_ratio (float): The number of warmup iterations.
        cooldown_ratio (float): The number of cooldown iterations.

    Returns:
        SequentialLR: The composite sequential learning rate scheduler.
    """
    assert warmup_ratio >= 0 and cooldown_ratio >= 0 and warmup_ratio + warmup_ratio <= 1.0
    warmup_iters = int(warmup_ratio * total_iters)
    cooldown_iters = int(cooldown_ratio * total_iters)

    warnings.filterwarnings("ignore", category=UserWarning, message=r'.*scheduler\.step\(\).*')

    warmup = lr_scheduler.LinearLR(optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters - 1)
    constant = lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=total_iters - warmup_iters - cooldown_iters)
    exponential = lr_scheduler.ExponentialLR(optimizer, gamma=end_factor ** (1 / cooldown_iters))

    return lr_scheduler.SequentialLR(optimizer,
                                     schedulers=[warmup, constant, exponential],
                                     milestones=[warmup_iters - 1, total_iters - cooldown_iters])


# --------- Numpy ---------

def smooth_line(x: np.ndarray, sigma=4.0, axis=-1, reduce_size: int = None):
    smooth_x = gaussian_filter1d(x, sigma=sigma, axis=axis)
    if reduce_size:
        assert 0 < reduce_size <= x.shape[axis], \
            f"reduce_size must be smaller or equal to than dimension size '{x.shape[axis]}'"
        idx = np.around(np.linspace(0, x.shape[axis] - 1, reduce_size)).astype(dtype=int)
        smooth_x = np.take(smooth_x, idx, axis=axis)
    return smooth_x


# --------- Pretty Printing Functions ---------

def pretty_number(n, d):
    """
    Convert a number to a readable format with the specified number of significant digits.

    Parameters:
    - n (float or int): The number to be formatted.
    - d (int): The number of significant digits desired in the output.

    Returns:
    - str: Formatted string representation of the number.

    Raises:
    - ValueError: If d is less than or equal to 0.
    """

    # Validate the input for number of significant digits
    if d <= 0:
        raise ValueError("Number of significant digits (d) should be greater than 0.")

    # If the number is 0, return "0" as output
    abs_n = abs(n)
    if abs_n == 0:
        return "0"

    # If the number is very small or very large, format it using scientific notation
    elif abs_n < 0.001 or abs_n >= 1e7:
        format_str = "{:." + str(d - 1) + "e}"
        return format_str.format(n)

    # For medium-sized numbers, format with fixed decimal places
    else:
        # Find out the number of digits before the decimal point
        int_digits = len(str(int(abs_n)))

        # Calculate how many digits should be after the decimal point
        decimal_digits = max(0, d - int_digits)

        # Construct the format string and return the formatted value
        format_str = "{:." + str(decimal_digits) + "f}"
        return format_str.format(n)


def tensor_dict_to_str(obj):
    """Converts all tensors in an object to their size strings."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor size to string
        return str(obj.size())
    elif isinstance(obj, dict):
        # Recursively convert tensors in dictionary
        return {k: tensor_dict_to_str(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        # Recursively convert tensors in list
        the_list = [tensor_dict_to_str(item) for item in obj]
        if len(the_list) > 3:
            the_list = the_list[:2] + ["..."] + the_list[-1:]
        return str(the_list)
    # Return the object unchanged if it's not a tensor, list or dictionary
    return obj


# --------- Argparse functions ---------

def valid_number(data_type, min_incl=None, min_excl=None, max_incl=None, max_excl=None):
    def validate_number(value):
        # Convert value based on data_type
        try:
            if data_type == int:
                num_value = int(value)
            elif data_type == float:
                num_value = float(value)
            else:
                raise ValueError("data_type must be either int or float.")
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid {data_type.__name__} value: {value}")

        # Check if both inclusive and exclusive bounds are set for min or max
        if (min_incl is not None and min_excl is not None) or \
           (max_incl is not None and max_excl is not None):
            raise ValueError("You cannot set both inclusive and exclusive bounds for min or max.")

        if (min_incl is not None and num_value < min_incl) or \
           (min_excl is not None and num_value <= min_excl) or \
           (max_incl is not None and num_value > max_incl) or \
           (max_excl is not None and num_value >= max_excl):

            bounds = []
            if min_incl is not None:
                bounds.append(f">= {min_incl}")
            if min_excl is not None:
                bounds.append(f"> {min_excl}")
            if max_incl is not None:
                bounds.append(f"<= {max_incl}")
            if max_excl is not None:
                bounds.append(f"< {max_excl}")
            bounds_str = " and ".join(bounds)

            raise argparse.ArgumentTypeError(f"{value} must be {bounds_str}.")

        return num_value

    return validate_number


# --------- Hyperopt ---------

def loguniform_10(label, a, b):
    return hp.loguniform(label, math.log(10 ** a), math.log(10 ** b))


# --------- Other utilities ---------

def merge_dictionaries(dictionaries):
    """Merge a list of dictionaries by appending values to common keys."""

    merged = defaultdict(list)

    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged[key].append(value)

    return dict(merged)


class ConsoleProgressBar:
    def __init__(self, total, display_interval=1):
        self.total = total
        self.current = 0
        self.display_interval = display_interval
        self.start_time = time.time()
        self.desc = ""

    def update(self, n=1, desc=None):
        if desc is not None:
            self.desc = desc

        self.current += n
        if self.current % self.display_interval == 0 or self.current == self.total:
            self.display()

    def display(self):
        elapsed_time = time.time() - self.start_time
        progress_percentage = (self.current / self.total) * 100

        # Estimation of total time required based on progress so far
        estimated_total_time = elapsed_time / (self.current / self.total) if self.current != 0 else 0
        estimated_time_left = estimated_total_time - elapsed_time

        estimated_end_time = time.localtime(time.time() + estimated_time_left)

        print(f"Progress: {self.current}/{self.total} ({progress_percentage:.2f}%) "
              f"[{self._format_time(elapsed_time)}<{self._format_time(estimated_time_left)}] "
              f"[ETA {time.strftime('%Y-%m-%d %H:%M', estimated_end_time)}] "
              f"[{self.desc}]")

    def finish(self):
        self.current = self.total
        self.display()

    @staticmethod
    def _format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


# --------- IO Functions ---------

def create_absolute_path(*components, create_directories: bool = True) -> str:
    """
    Creates an absolute path using pathlib and optionally creates missing directories.

    :param components: Components of the path to be joined.
    :param create_directories: If True, missing directories along the path will be created.
    :return: The created absolute path.
    """
    absolute_path = Path(*components).resolve()

    if create_directories and not absolute_path.is_file():
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

    return str(absolute_path)


def read_json(filepath: str) -> dict:
    """Read a JSON file and return its contents as a dictionary."""
    with open(filepath, "r") as file:
        return json.load(file)


def write_json(filepath: str, data: dict) -> None:
    """Write dictionary data as JSON to the specified file path."""
    with open(filepath, "w") as file:
        json.dump(data, file, indent=4)


def read_bz2_file(filepath, default_value=None) -> dict:
    """Read a compressed JSON file and return its contents as a dictionary.
       If the file doesn't exist, return the default value."""
    if os.path.exists(filepath):
        with bz2.BZ2File(filepath, 'r') as f:
            return json.loads(f.read())
    return default_value


def write_bz2_file(filepath, data: dict):
    """Write dictionary data as compressed JSON to the specified file path."""
    with bz2.BZ2File(filepath, 'w') as bz2_file:
        bz2_file.write(bytes(json.dumps(data, indent=4), encoding='latin1'))


def write_module(module, filepath):
    """
    Write a PyTorch nn.Module to a file.

    Arguments:
    module -- The nn.Module object to be saved.
    filepath -- The path of the file to save the module to.
    """
    torch.save(module.state_dict(), filepath)


def read_module(module_class, filepath):
    """
    Read a PyTorch nn.Module from a file.

    Arguments:
    module_class -- The class of the nn.Module to be loaded.
    filepath -- The path of the file to load the module from.

    Returns:
    module -- The loaded nn.Module object.
    """
    module = module_class()
    module.load_state_dict(torch.load(filepath))
    return module


def read_pickled_object(filepath: str, default_value=None):
    """
    Reads a pickled object from a file and returns it.

    Args:
        filepath: A string representing the path to the pickled file.
        default_value: A default value to return if the file doesn't exist.

    Returns:
        The loaded object from the pickled file.
        If the file doesn't exist and a default value is provided, the default value is returned.
        If the file doesn't exist and no default value is provided, None is returned.
    """
    if os.path.exists(filepath):
        with open(filepath, "rb") as pickle_file:
            obj = pickle.load(pickle_file)
            return obj
    return default_value


def write_object_pickled(filepath: str, data):
    """
    Writes an object to a file in pickled format.

    Args:
        filepath: A string representing the path to the pickled file to write.
        data: An object to write to the file.
    """
    with open(filepath, "wb") as pickle_file:
        pickle.dump(data, pickle_file)