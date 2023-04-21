import bz2
import json
import math
import os
import pickle
from datetime import datetime
from typing import TypeVar, List
from torch import nn
from tqdm import trange

_T = TypeVar('_T')


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
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        # count all parameters, including non-trainable parameters
        total_params = sum(p.numel() for p in model.parameters())

    return total_params


def share_parameters(default_model: nn.Module, *model_list: nn.Module) -> None:
    """
    Shares the parameter tensors of a default model with a list of PyTorch models.

    Args:
        default_model: A PyTorch model whose parameters will be shared with other models.
        *model_list: A variable number of PyTorch models to share parameters with.

    Raises:
        ValueError: If the list of models is empty.

    Note:
        This function assumes that all models have the same architecture, i.e., the same
        set of named parameters. If the models have different architectures, sharing their
        parameters may result in unexpected behavior.
    """
    if len(model_list) == 0:
        raise ValueError("model_list must not be empty")

    # iterate over the named parameters of the default model
    for name, param in default_model.named_parameters():
        # share the parameter tensor between all the models in the list
        name = name.split('.')
        for model in model_list:
            target_param = model
            for n in name[:-1]:
                target_param = getattr(target_param, n)
            setattr(target_param, name[-1], param)


def join_path(*sub_paths, make_dirs: bool = True) -> str:
    """
    Joins a set of sub-paths into a single absolute path, creating directories along the way if necessary.

    Args:
        *sub_paths: A variable number of string paths to join together.
        make_dirs: A boolean flag indicating whether to create directories for any missing path components.
            If True, directories will be created as needed.

    Returns:
        A string representing the absolute path of the joined sub-paths.
    """
    # join the sub-paths together into a single absolute path
    full_path = os.path.abspath(os.path.join(*sub_paths))

    if make_dirs:
        # create any missing directories along the path
        is_file = len(os.path.splitext(full_path)[-1]) > 0
        folder = os.path.split(full_path)[0] if is_file else full_path
        if not os.path.exists(folder):
            os.makedirs(folder)

    # return the full path
    return full_path


def load_json_file(filepath: str) -> dict:
    """
    Reads a JSON file and returns the contents as a dictionary.

    Args:
        filepath: A string representing the path to the JSON file.

    Returns:
        A dictionary containing the contents of the JSON file.
    """
    with open(filepath, "r") as json_file:
        return json.load(json_file)


def save_json_file(filepath: str, data: dict) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        filepath: A string representing the path to the JSON file to write.
        data: A dictionary containing the data to write to the file.
    """
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=4)


def load_bz2_file(filepath, default_value=None) -> dict:
    """
    Reads a JSON data from a compressed .bz2 file and returns it as a dictionary.

    Args:
        filepath: A string representing the path to the .bz2 file.
        default_value: A default value to return if the file doesn't exist.

    Returns:
        A dictionary containing the contents of the .bz2 file.
        If the file doesn't exist and a default value is provided, the default value is returned.
        If the file doesn't exist and no default value is provided, None is returned.
    """
    if os.path.exists(filepath):
        with bz2.BZ2File(filepath, 'r') as f:
            return json.loads(f.read())
    return default_value


def save_bz2_file(filepath, data: dict):
    """
    Writes a dictionary to a compressed .bz2 file.

    Args:
        filepath: A string representing the path to the .bz2 file to write.
        data: A dictionary containing the data to write to the file.
    """
    with bz2.BZ2File(filepath, 'w') as bz2_file:
        bz2_file.write(bytes(json.dumps(data, indent=4), encoding='latin1'))


def load_pickled_object(filepath: str, default_value=None):
    """
    Loads a pickled object from a file and returns it.

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


def save_object_pickled(filepath: str, data):
    """
    Saves an object to a file in pickled format.

    Args:
        filepath: A string representing the path to the pickled file to write.
        data: An object to write to the file.
    """
    with open(filepath, "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def short_scientific_notation(x, decimals=2):
    """
    Returns a string representation of the given number in scientific notation,
    with the specified number of decimal places and the shortest possible string length.

    Args:
        x: A number to convert to scientific notation.
        decimals: The number of decimal places to include in the output.

    Returns:
        A string representation of the number in scientific notation.
    """
    if x == 0:
        return '0'
    # Use the format() function to get the number in scientific notation with the specified number of decimal places
    formatted_num = "{:.{}e}".format(x, decimals)

    # Split the formatted number into the mantissa and exponent parts
    mantissa, exponent = formatted_num.split('e')

    # Remove any trailing zeros and the decimal point from the mantissa
    mantissa = mantissa.rstrip('0').rstrip('.')

    # If the exponent is negative, add a minus sign to the front and remove the leading zero
    if exponent[0] == '-':
        exponent_str = "-" + exponent[1:].lstrip('0')
    else:
        exponent_str = exponent.lstrip('+').lstrip('0')

    # Combine the mantissa and exponent parts into the final string
    return mantissa + 'e' + exponent_str


def short_number(x, significant_numbers=3):
    """
    Converts a number to a short string representation with a maximum number of significant digits.

    Args:
        x (float): The number to be converted.
        significant_numbers (int): The maximum number of significant digits.

    Returns:
        A string representation of the number with a maximum of `num_sig_digits` significant digits,
        in either decimal or scientific notation (with a mantissa of 1 digit) and the shortest possible string length.
    """
    if x == 0:
        return '0'

    # Save the default string representation of x
    default_str = str(x)

    # Round x to the desired number of significant digits
    num_left_digits = max(0, significant_numbers - int(math.floor(math.log10(abs(x)))) - 1)
    rounded_x = round(x, num_left_digits)

    # Choose the shortest string representation of x
    return min([default_str, rounded_x, short_scientific_notation(x, 2)], key=len)


def time_str(delta):
    return str(delta).split('.')[0]


class ProgressWriter:
    def __init__(self, total: int, desc: str = None):
        self.total = total
        self.counter = 0
        self.desc = desc + ' ' if desc is not None else ''
        self.timestamps = [datetime.now()]
        self.interval = max(2, int(round(self.total / 20)))
        print(f"[0/{total}, 0%, 0:00:00<?, ?/it]", flush=True)
        self.last_print = self.timestamps[-1]

    def reset(self, total: int, desc: str = ''):
        self.__init__(total, desc)

    def update(self):
        self.counter += 1
        self.timestamps.append(datetime.now())

        if self.counter == self.total or (self.timestamps[-1] - self.last_print).total_seconds() > 1:
            total = self.timestamps[-1] - self.timestamps[0]
            inverval_timestamps = self.timestamps[-self.interval:]
            time_per_iter = (inverval_timestamps[-1] - inverval_timestamps[0]) / (len(inverval_timestamps) - 1)
            predicted = time_per_iter * (self.total - self.counter)
            print(f"{self.desc}["
                  f"{self.counter}/{self.total}, "
                  f"{round(100*self.counter/self.total)}%, "
                  f"{time_str(total)}<{time_str(predicted)}, "
                  f"{time_str(time_per_iter)}/it"
                  f"]", flush=True)
            self.last_print = self.timestamps[-1]


def tqdm_print(data, desc=None):
    progress_writer = ProgressWriter(total=len(data), desc=desc)
    for d in data:
        yield d
        progress_writer.update()


def trange_print(total: int, desc=None):
    progress_writer = ProgressWriter(total=total, desc=desc)
    for _i in range(total):
        yield _i
        progress_writer.update()


def trange_mode(total: int, mode: str, desc: str = None):
    if mode == 'tqdm':
        for i in trange(total, desc=desc):
            yield i
    elif mode == 'console':
        progress_writer = ProgressWriter(total=total, desc=desc)
        for i in range(total):
            yield i
            progress_writer.update()
    else:
        raise NotImplementedError(f"mode '{mode}' is not supported.")


if __name__ == '__main__':
    for i in [234876234, 0.00002034, 12312.1892439234239, 0, 0.1]:
        print(short_scientific_notation(i, 3))
        print(short_scientific_notation(i, 3))
