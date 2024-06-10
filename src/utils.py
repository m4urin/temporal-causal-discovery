import argparse
import bz2
import json
import math
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Union, Iterable

import torch
from torch import nn
from torch.optim import lr_scheduler
import mlflow
from tqdm import trange
import numpy as np
from collections import defaultdict
from hyperopt import hp
from hyperopt.pyll import scope
from scipy.ndimage import gaussian_filter1d


# --------- Paths -------

""" Get the full paths"""
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

""" GPULab """
GPULAB_JOB_ID = None
if 'GPULAB_JOB_ID' in os.environ:
    GPULAB_JOB_ID = os.environ['GPULAB_JOB_ID'][:6]
    OUTPUT_DIR = os.path.join(os.path.split(PROJECT_ROOT)[0], 'outputs')
else:
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
os.makedirs(TEST_DIR, exist_ok=True)

""" MLFlow """
TRACKING_URI = f"file:///{os.path.join(OUTPUT_DIR, 'mlruns')}"
mlflow.set_tracking_uri(TRACKING_URI)

if __name__ == '__main__':
    print(f'PROJECT_ROOT: {PROJECT_ROOT}')
    print(f'DATA_DIR: {DATA_DIR}')
    print(f'OUTPUT_DIR: {OUTPUT_DIR}')
    print(f'TEST_DIR: {TEST_DIR}')
    print(f'MLFlow URI: {TRACKING_URI}')


# --------- TCN ---------

def receptive_field(n_blocks, n_layers, kernel_size, **kwargs):
    return n_layers * ((2 ** n_blocks) - 1) * (kernel_size - 1) + 1


def generate_architecture_options(max_lags, marge, minimum_num_options=1,
                                  n_blocks=None, n_layers_per_block=None, kernel_size=None):
    """Generate architecture options based on given parameters."""

    if n_blocks and n_layers_per_block and kernel_size:
        assert n_blocks >= 1 and n_layers_per_block >= 1 and kernel_size >= 2
        return [{'n_blocks': n_blocks, 'n_layers_per_block': n_layers_per_block, 'kernel_size': kernel_size}]

    assert max_lags is not None and max_lags >= 2, \
        'if --n_block, --n_layers_per_block, and --kernel_size are not/partially provided, ' \
        'a --max_lags must be provided to establish a possible architecture.'

    blocks_range = (1, 100) if n_blocks is None else (n_blocks, n_blocks + 1)
    layers_range = (1, 2 + 1) if n_layers_per_block is None else (n_layers_per_block, n_layers_per_block + 1)
    kernel_range = (2, 1000) if kernel_size is None else (kernel_size, kernel_size + 1)

    layers_min = layers_range[0]
    kernel_min = kernel_range[0]

    result = set()
    current_marge = marge
    while len(result) < minimum_num_options:
        for blocks in range(*blocks_range):
            if max_lags + current_marge > receptive_field(blocks, layers_min, kernel_min) >= max_lags:
                result.add((blocks, layers_min, kernel_min))
                break
            for layers in range(*layers_range):
                if max_lags + current_marge > receptive_field(blocks, layers, kernel_min) >= max_lags:
                    result.add((blocks, layers, kernel_min))
                    break
                for k_size in range(*kernel_range):
                    if max_lags + current_marge > receptive_field(blocks, layers, k_size) >= max_lags:
                        result.add((blocks, layers, k_size))
                        break
        if current_marge > 3 * marge:
            break
        current_marge += 1

    valids = list(result)
    if len(valids) == 0:
        err = f"Cannot find a valid architecture for max_lags={max_lags}"
        if n_blocks:
            err += f", n_blocks={n_blocks}"
        if n_layers_per_block:
            err += f", n_blocks={n_layers_per_block}"
        if kernel_size:
            err += f", n_blocks={kernel_size}"
        err += f". Even after increasing the marge to 3 x marge={3 * marge}."
        raise ValueError(err)

    return [{'n_blocks': b, 'n_layers_per_block': l, 'kernel_size': k} for b, l, k in valids]


# --------- PyTorch Functions ---------

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """ Counts the number of parameters in a PyTorch model. """
    if trainable_only:
        # count only trainable parameters
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # count all parameters, including non-trainable parameters
        return sum(p.numel() for p in model.parameters())


def min_max_normalization(x: torch.Tensor, min_val=0.0, max_val=1.0):
    # Find the minimum and maximum values along the specified dimension
    min_x = x.min()
    max_x = x.max()

    # Normalize the tensor
    x_normalized = (x - min_x) / (max_x - min_x)

    # Scale to the new min and max
    x_normalized = x_normalized * (max_val - min_val) + min_val

    return x_normalized


def weighted_mean(x: torch.Tensor, weights: torch.Tensor, dim, keepdim=False):
    """ Calculate the weighted mean along a specified dimension using given weights. """
    return (weights * x).sum(dim=dim, keepdim=keepdim) / weights.sum(dim=dim, keepdim=keepdim)


def weighted_std(x: torch.Tensor, weights: torch.Tensor, dim, keepdim=False):
    """ Compute the weighted standard deviation along a specified dimension using provided weights and input tensor. """
    total_weight = weights.sum(dim=dim, keepdim=True)
    weighted_mean = (weights * x).sum(dim=dim, keepdim=True) / total_weight
    if not keepdim:
        total_weight = total_weight.squeeze(dim)
    weighted_variance = (weights * (x - weighted_mean).pow(2)).sum(dim=dim, keepdims=keepdim) / total_weight
    return torch.sqrt(weighted_variance)


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


def get_module_device(model: nn.Module):
    """ Obtains the device of the model. """
    return next(model.parameters()).device


def to_cuda(data):
    if isinstance(data, dict):
        return {k: to_cuda(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.contiguous().cuda()
    else:
        return data


def to_cpu(data):
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu()
    else:
        return data


def detach(data):
    if isinstance(data, dict):
        return {k: detach(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.detach()
    else:
        return data


def augment_with_sine(x: torch.Tensor):
    batch_size, n, seq_len = x.size()

    # Create a linear space from 0 to 2*pi for one complete cycle
    t = torch.linspace(0, 2 * np.pi, seq_len, device=x.device)

    # Compute sine and cosine
    sine_wave = torch.sin(t)
    cosine_wave = torch.cos(t)

    # Stack the two waves together to get shape (2, seq)
    combined_wave = torch.stack([sine_wave, cosine_wave])

    # Reshape to get shape (bs, n, 2, seq)
    combined_wave = combined_wave.unsqueeze(0).unsqueeze(0).expand(batch_size, n, -1, -1)
    # Reshape to get shape (bs, n, 1, seq)
    x = x.reshape(batch_size, n, 1, seq_len)
    # Cat to get shape (bs, n, 3, seq)
    x = torch.cat((x, combined_wave), dim=2)
    # Reshape to get shape (bs, n*3, seq)
    return x.reshape(batch_size, -1, seq_len)


# --------- Numpy ---------

def smooth_line(x: np.ndarray, sigma=4.0, axis=-1, reduce_size: int = None):
    """ Smooth a one-dimensional array using Gaussian filtering with optional dimension reduction. """
    smooth_x = gaussian_filter1d(x, sigma=sigma, axis=axis)
    if reduce_size:
        assert 0 < reduce_size <= x.shape[axis], \
            f"reduce_size must be smaller or equal to than dimension size '{x.shape[axis]}'"
        idx = np.around(np.linspace(0, x.shape[axis] - 1, reduce_size)).astype(dtype=int)
        smooth_x = np.take(smooth_x, idx, axis=axis)
    return smooth_x


def to_numpy(x: Union[torch.Tensor, np.ndarray, Iterable, dict]) -> Union[np.ndarray, dict[np.ndarray]]:
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# --------- Pretty Printing Functions ---------

def pretty_number(a_number, significant_digits=1):
    """ Convert a number to a readable format with the specified number of significant digits. """
    # Validate the input for number of significant digits
    if significant_digits <= 0:
        raise ValueError("Number of significant digits (d) should be greater than 0.")

    # If the number is 0, return "0" as output
    abs_n = abs(a_number)
    if abs_n == 0:
        return "0"

    # If the number is very small or very large, format it using scientific notation
    format1 = str(a_number)
    format2 = ("{:." + str(significant_digits) + "e}").format(a_number)

    if len(format1) < len(format2):
        return format1
    return format2


def tensor_dict_to_str(obj):
    """Converts all tensors in an object to their size strings."""
    if isinstance(obj, torch.Tensor):
        # Convert tensor size to string
        return str(obj.size())
    elif isinstance(obj, dict):
        # Recursively convert tensors in dictionary
        return {k: tensor_dict_to_str(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        # Recursively convert tensors in list
        the_list = [tensor_dict_to_str(item) for item in obj]
        if len(the_list) > 3:
            the_list = the_list[:2] + ["..."] + the_list[-1:]
        if isinstance(obj, tuple):
            the_list = tuple(the_list)
        return str(the_list)
    # Return the object unchanged if it's not a tensor, list or dictionary
    return obj


# --------- Argparse functions ---------

def valid_number(data_type, min_incl=None, min_excl=None, max_incl=None, max_excl=None):
    """
    Generate a validation function for a specified data type with optional bounds checks,
    supporting both inclusive and exclusive ranges.
    """

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

def hp_loguniform_10(label, low, high):
    """
    A loguniform distribution with base 10 for use in hyperparameter
    optimization, bounded between 10^low and 10^high.
    """
    return hp.loguniform(label, math.log(10 ** low), math.log(10 ** high))


def hp_pow(label, low: int, high: int, base: int = 2):
    """
    Create a power distribution for hyperparameter optimization.
    Example:
        If base=2, low=2, and high=5, the distribution will include values: [2^2, 2^3, 2^4, 2^5].
    """
    return scope.pow(base, scope.int(hp.quniform(label, low, high, 1.0)))


# --------- Other utilities ---------

def merge_dictionaries(dictionaries):
    """Merge a list of dictionaries by appending values to common keys."""
    merged = defaultdict(list)

    for dictionary in dictionaries:
        for key, value in dictionary.items():
            merged[key].append(value)

    return dict(merged)


def measure_function_times(functions, input_args, num_iterations, num_trials, num_cache_trials=2, function_names=None,
                           show_progress=True):
    """
    Measure the execution time of a set of functions over multiple trials and iterations,
    providing mean time in milliseconds per iteration. Supports caching and shuffling of function order.
    """
    if function_names is None:
        function_names = [f"Function {i}" for i in range(len(functions))]

    max_name_length = max(len(name) for name in function_names)
    padded_function_names = [name.ljust(max_name_length) for name in function_names]

    time_records = np.zeros((len(functions), num_trials))
    indexed_functions = list(enumerate(functions))

    progress_bar = trange(num_cache_trials + num_trials, disable=not show_progress)

    for _ in range(num_cache_trials):
        for _, func in indexed_functions:
            for _ in range(num_iterations):
                func(*input_args)
        np.random.shuffle(indexed_functions)
        progress_bar.update()

    for i_trial in range(num_trials):
        for i, func in indexed_functions:
            t0 = time.time()
            for _ in range(num_iterations):
                func(*input_args)
            t1 = time.time()
            time_records[i, i_trial] = t1 - t0
        np.random.shuffle(indexed_functions)
        progress_bar.update()
    progress_bar.close()

    for name, times in zip(padded_function_names, time_records * 1000 / num_iterations):
        mean_time = np.mean(times)
        print(f"{name}: {mean_time:.2e} ms/it")


class ConsoleProgressBar:
    """
    Create a console-based progress bar for tracking the advancement of a process.
    Displays progress percentage, elapsed and estimated time, along with an optional description.
    """

    def __init__(self, total, display_interval=1, title=None):
        self.total = total
        self.current = 0
        self.display_interval = display_interval
        self.start_time = time.time()
        self.desc = ""
        self.title = title if title is not None else "Progress"
        self.update(0)

    def update(self, n=1, desc=None):
        if desc is not None:
            self.desc = desc

        self.current += n
        if self.current % self.display_interval == 0 or self.current == self.total:
            self.display()

    def set_description(self, desc):
        self.desc = desc

    def display(self):
        elapsed_time = time.time() - self.start_time
        progress_percentage = (self.current / self.total) * 100

        # Estimation of total time required based on progress so far
        estimated_total_time = elapsed_time / (self.current / self.total) if self.current != 0 else 0
        estimated_time_left = estimated_total_time - elapsed_time

        estimated_end_time = time.localtime(time.time() + estimated_time_left)

        eta_str = time.strftime('%Y-%m-%d %H:%M', estimated_end_time) if self.current > 0 else "???"
        print(f"{self.title}: {self.current}/{self.total} ({progress_percentage:.2f}%) "
              f"[{self._format_time(elapsed_time)}<{self._format_time(estimated_time_left)}] "
              f"[ETA {eta_str}] "
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
    """ Creates an absolute path using pathlib and optionally creates missing directories. """
    absolute_path = Path(*components).resolve()

    if create_directories and not absolute_path.is_file():
        absolute_path.parent.mkdir(parents=True, exist_ok=True)

    return str(absolute_path)


def read_json(filepath: str) -> dict:
    """Read a JSON file and return its contents as a dictionary."""
    with open(filepath, "r") as file:
        return json.load(file)


def write_json(filepath: str, data: dict):
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
        bz2_file.write(bytes(json.dumps(data), encoding='latin1'))


def write_module(module, filepath):
    """ Write a PyTorch nn.Module to a file.  """
    torch.save(module.state_dict(), filepath)


def read_module(module_class, filepath):
    """ Read a PyTorch nn.Module from a file. """
    module = module_class()
    module.load_state_dict(torch.load(filepath))
    return module


def read_pickled_object(filepath: str, default_value=None):
    """ Reads a pickled object from a file and returns it. """
    if os.path.exists(filepath):
        with open(filepath, "rb") as pickle_file:
            obj = pickle.load(pickle_file)
            return obj
    return default_value


def write_object_pickled(filepath: str, data):
    """ Writes an object to a file in pickled format. """
    with open(filepath, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


# --------- CauseMe benchmark ---------

def get_method_simple_description(model, weight_sharing=False, recurrent=False,
                                  aleatoric=False, epistemic=False, softmax_method=None, **args):
    sub_models = [model]
    if weight_sharing:
        sub_models.append('WS')
    if recurrent:
        sub_models.append('Rec')
    if epistemic:
        sub_models.append('E')
    elif aleatoric:
        sub_models.append('A')
    if model == 'TAMCaD' and softmax_method is not None:
        sub_models.append(softmax_method)
    return "-".join(sub_models)


def get_method_hp_description(model_str, hidden_dim, lambda1, architecture,
                              n_heads, softmax_method, recurrent=False, weight_sharing=False, uncertainty_aware=None,
                              **args):
    kernel_size = architecture['kernel_size']
    n_layers_per_block = architecture['n_layers_per_block']
    n_blocks = architecture['n_blocks']

    if model_str == 'NAVAR':
        method_sha = "e0ff32f63eca4587b49a644db871b9a3"
    elif model_str == 'TAMCaD':
        method_sha = "8fbf8af651eb4be7a3c25caeb267928a"
    else:
        raise ValueError("not a valid model")

    result = []
    if uncertainty_aware:
        result.append('UA')
    result = "-".join(result)

    params = f""
    if model_str == 'TAMCaD':
        params += f"n_heads={n_heads}, "  # beta={pretty_number(beta)}, "  {softmax_method},
    params += f"dim={hidden_dim}, b_n_k=({n_blocks},{n_layers_per_block},{kernel_size})"  # layers_per_block={n_layers_per_block}, " \
    # f"kernel_size={kernel_size}, lambda1={pretty_number(lambda1)}"
    # f"epochs={epochs}, lr={pretty_number(lr)}, " \
    # f"weight_decay={pretty_number(weight_decay)}, dropout={pretty_number(dropout)}"

    if len(result) > 0:
        result += ', ' + params
    else:
        result = params
    return {'parameter_values': result, 'method_sha': method_sha}
