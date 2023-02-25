import bz2
import json
import math
import os
import pickle
from datetime import datetime
from typing import TypeVar
from torch import nn
from tqdm import trange

_T = TypeVar('_T')


def count_params(model: nn.Module, count_trainable_only=False):
    if count_trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def join(*sub_paths, make_dirs=True):
    full_path = os.path.abspath(os.path.join(*sub_paths))
    is_file = len(os.path.splitext(full_path)[-1]) > 0
    if make_dirs:
        folder = os.path.split(full_path)[0] if is_file else full_path
        if not os.path.exists(folder):
            os.makedirs(folder)
    return full_path


def read_json(filename) -> dict:
    with open(filename, "r") as f:
        return json.load(f)


def write_json(filename, data: dict):
    with open(filename, "w") as f:
        return json.dump(data, f, indent=4)


def read_bz2(filename, default=None) -> dict:
    if os.path.exists(filename):
        with bz2.BZ2File(filename, 'r') as f:
            return json.loads(f.read())
    return default


def write_bz2(filename, data: dict):
    with bz2.BZ2File(filename, 'w') as mybz2:
        mybz2.write(bytes(json.dumps(data, indent=4), encoding='latin1'))


def read_pickled_object(filename, default: _T = None) -> _T:
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            obj: _T = pickle.load(f)  # load checkpoint
            return obj
    return default


def write_object_pickled(filename, data: _T):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def e_format(x, decimals: int = 2):
    a, b = ("{:." + str(decimals) + "e}").format(x).split('e')
    a = a.rstrip('0').rstrip('.')
    b, c = b[0] if b[0] == '-' else '', b[1:].lstrip('0')
    if len(c) == 0:
        return a
    return a + 'e' + b + c


def pretty_number(x, significant_numbers: int = 3):
    if x == 0:
        return '0'
    default = str(x)
    x = round(x, max(0, significant_numbers - int(math.floor(math.log10(abs(x)))) - 1))
    return min([default, str(x), e_format(x)], key=len)


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
