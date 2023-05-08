import math
import numbers

import torch

from src.data.dataset import Dataset


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


def short_number(x: numbers.Number, significant_numbers=4):
    """
    Converts a number to a short string representation with a maximum number of significant digits.

    Args:
        x (Number): The number to be converted.
        significant_numbers (int): The maximum number of significant digits.

    Returns:
        A string representation of the number with a maximum of `num_sig_digits` significant digits,
        in either decimal or scientific notation (with a mantissa of 1 digit) and the shortest possible string length.
    """
    if x == 0:
        return '0'

    # Save the default string representation of x
    default_str = str(x)
    x = float(x)

    # Round x to the desired number of significant digits
    num_left_digits = max(0, significant_numbers - int(math.floor(math.log10(abs(x)))) - 1)
    rounded_x = str(round(x, num_left_digits))

    # Choose the shortest string representation of x
    result = min([default_str, rounded_x, short_scientific_notation(x, 2)], key=len)

    if result.endswith('.0'):
        return result[:-2]
    return result


def dict_to_url(d: dict):
    d_str = {}
    for k in list(d.keys()):
        v = d[k]
        if isinstance(v, numbers.Number):
            d_str[k] = short_number(v)
        elif isinstance(v, type):
            d_str[k] = v.__name__
        elif isinstance(v, Dataset):
            if v.name is not None:
                d_str['experiment'] = v.name
            d_str['batch_size'] = v.batch_size
            d_str['num_variables'] = v.num_variables
            d_str['sequence_length'] = v.sequence_length
        elif isinstance(v, torch.Tensor):
            d_str[k] = str(v.size()).split('.')[1]

    return ", ".join([f"{k}={v}" for k, v in d_str.items()])
