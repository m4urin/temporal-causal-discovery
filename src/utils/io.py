import bz2
import json
import os
import pickle


def join_paths(*sub_paths, make_dirs: bool = True) -> str:
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
