from collections.abc import Collection
from typing import Iterable, Union, Generator, Any
from math import prod
from time import time

import numpy as np
from tqdm import trange


class Nested_trange:
    """
    NestedProgressWriter is a wrapper around 'trange' that provides a way to display nested progress bars with descriptions.
    """

    def __init__(self, loops: Iterable[tuple[Union[int, Collection], str]], refresh_rate: float = 0.01):
        """
        Initialize the NestedProgressWriter object.

        Args:
            loops (Iterable[tuple[int or Collection, str]]): An iterable containing tuples of iteration counts
                and descriptions for each level of nested loops.
        """
        self._iters = [range(iterable) if isinstance(iterable, int) else iterable for iterable, _ in loops]
        self._lengths = np.array([len(iterable) for iterable in self._iters], dtype=int)
        self._descriptions = [desc for _, desc in loops]

        cum_prod = np.flip(np.cumprod(np.flip(self._lengths)))
        _totals = np.empty(len(self._lengths), dtype=int)
        _totals[-1] = 1
        _totals[:-1] = cum_prod[1:]

        self._pbar = trange(cum_prod[0])
        self._totals = _totals

        self.refresh_rate = refresh_rate
        self._last_desc_update = time()

        self._info = None
        self._n = np.zeros(len(self._lengths), dtype=int)

        self.refresh_descriptions()

    def set_loop_description(self, loop_index, desc: str, refresh=True):
        """
        Set the description for a specific loop at the given index.

        Args:
            loop_index (int): The index of the loop for which to set the description.
            desc (str): The new description for the loop.
            refresh (bool, optional): Whether to refresh the descriptions after setting the description.
                Defaults to True.
        """
        self._descriptions[loop_index] = desc
        if refresh:
            self.refresh_descriptions()

    def set_info(self, info: str, refresh=False):
        """
        Set additional information to be displayed with the descriptions.

        Args:
            info (str): The additional information to be displayed.
            refresh (bool, optional): Whether to refresh the descriptions after setting the information.
                Defaults to False.
        """
        self._info = info
        if refresh:
            self.refresh_descriptions()

    def refresh_descriptions(self):
        """
        Refresh the descriptions displayed in the progress bar.
        """
        n = self._pbar.n
        if n == self._pbar.total:
            indices = self._lengths - 1
        else:
            indices = [0] * len(self._lengths)
            for i in reversed(range(len(self._lengths))):
                indices[i] = n % self._lengths[i]
                n //= self._lengths[i]

        full_desc = [f'{desc} [{i + 1}/{total}]' for i, total, desc in zip(indices, self._lengths, self._descriptions)]
        full_desc = ", ".join(full_desc)
        if self._info is not None:
            full_desc = f"{full_desc} {self._info}"

        self._pbar.set_description(full_desc)

    def update(self, loop_index: int):
        self._n[loop_index] += 1
        self._n[loop_index + 1:] = 0
        self._pbar.update(n=np.sum(self._totals * self._n) - self._pbar.n)

        t = time()
        if t - self._last_desc_update > self.refresh_rate or self._pbar.n == self._pbar.total:
            self.refresh_descriptions()
            self._last_desc_update = t

    def iter(self, loop_index) -> Generator[Any, Any, None]:
        """
        Iterate over the elements of a specific loop.

        Args:
            loop_index (int): The index of the loop to iterate over.

        Yields:
            The elements of the loop.
        """
        for x in self._iters[loop_index]:
            yield x
            self.update(loop_index)

    def finish(self):
        self._pbar


if __name__ == '__main__':
    from time import sleep
    from random import random

    models = [f'Model{i + 1}' for i in range(5)]
    settings = [(models, 'Model'), (3, 'Hyperopt'), (100, 'Epoch')]

    pbar = Nested_trange(settings)

    for i, model in enumerate(pbar.iter(loop_index=0)):
        pbar.set_loop_description(0, desc=model)
        if i >= 2:
            continue
        for j in range(3):
            for k in pbar.iter(loop_index=2):
                sleep(0.01)
                rand = random()
                pbar.set_info(f'Loss={round(rand, 3)}')
                if rand < 0.05:
                    break
            pbar.update(loop_index=1)
