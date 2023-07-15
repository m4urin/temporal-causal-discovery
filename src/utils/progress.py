from datetime import datetime, timedelta
from typing import Generator, Iterable, Sized

import enlighten
from tqdm import trange

from definitions import GPULAB_JOB_ID


class ProgressWriter:
    def __init__(self, total: int, desc: str = None):
        """
        Initializes the ProgressWriter object.

        Parameters:
        -----------
        total : int
            The total number of iterations to complete.
        desc : str, optional
            Description of the progress updates.
        """
        self.total = total
        self.counter = 0
        self.desc = desc + ' ' if desc is not None else ''
        self.timestamps = [datetime.now()]
        self.interval = max(2, int(round(self.total / 20)))
        print(f"{self.desc}[0/{total}, 0%, 0:00:00<?, ?/it]", flush=True)
        self.last_print = self.timestamps[-1]

    def reset(self, total: int, desc: str = None):
        """
        Resets the ProgressWriter object with a new total and description.

        Parameters:
        -----------
        total : int
            The new total number of iterations to complete.
        desc : str, optional
            The new description of the progress updates.
        """
        self.__init__(total, desc)

    def update(self):
        """
        Updates the ProgressWriter object with the current iteration count and displays a progress update if necessary.
        """
        self.counter += 1
        self.timestamps.append(datetime.now())

        if self.counter == self.total or (self.timestamps[-1] - self.last_print).total_seconds() > 1:
            total = self.timestamps[-1] - self.timestamps[0]
            interval_timestamps = self.timestamps[-self.interval:]
            time_per_iter = (interval_timestamps[-1] - interval_timestamps[0]) / (len(interval_timestamps) - 1)
            predicted = time_per_iter * (self.total - self.counter)
            print(f"{self.desc}["
                  f"{self.counter}/{self.total}, "
                  f"{round(100 * self.counter / self.total)}%, "
                  f"{time_to_str(total)}<{time_to_str(predicted)}, "
                  f"{time_to_str(time_per_iter)}/it"
                  f"]", flush=True)
            self.last_print = self.timestamps[-1]


def time_to_str(delta: timedelta) -> str:
    """
    Convert a timedelta object to a string representation.

    Args:
        delta: A timedelta object representing the difference between two times.

    Returns:
        A string representing the timedelta object, rounded down to the nearest second.
    """
    return str(delta).split('.')[0]


def tqdm_print(data: list, desc: str = None) -> Generator:
    """
    A generator function that yields items from a list while displaying progress updates.

    Parameters:
    -----------
    data : list
        The list of items to iterate over.
    desc : str, optional
        Description of the progress updates.

    Returns:
    --------
    A generator that yields items from the list while displaying progress updates.
    """
    progress_writer = ProgressWriter(total=len(data), desc=desc)
    for d in data:
        yield d
        progress_writer.update()


def trange_print(total: int, desc: str = None) -> Generator:
    """
    A generator function that yields numbers from 0 to `total` while displaying progress updates.

    Parameters:
    -----------
    total : int
        The maximum number to iterate up to (exclusive).
    desc : str, optional
        Description of the progress updates.

    Returns:
    --------
    A generator that yields numbers from 0 to `total` while displaying progress updates.
    """
    progress_writer = ProgressWriter(total=total, desc=desc)
    for i in range(total):
        yield i
        progress_writer.update()


USE_CONSOLE = GPULAB_JOB_ID is not None


def use_console(flag: bool):
    global USE_CONSOLE
    USE_CONSOLE = flag


def iter_with_progress(total: int, show_progress=True, desc: str = None) -> Generator:
    """
    A generator function that yields numbers from 0 to `total` while displaying progress updates.

    Args:
        total : int
            The maximum number to iterate up to (exclusive).
        desc : str, optional
            Description of the progress updates.
        show_progress: TODO

    Returns:
        A generator that yields numbers from 0 to `total` while displaying progress updates.
    """
    if show_progress:
        if USE_CONSOLE:
            for i in trange_print(total=total, desc=desc):
                yield i
        else:
            for i in trange(total, desc=desc):
                yield i
    else:
        for i in range(total):
            yield i


def iter_batched(data, max_batch_size: int, show_progress: bool = False, desc: str = None):
    """
    This function takes an iterable of data and splits it into batches of a given size.
    """
    # Calculate the number of iterations required to process all data
    iterations = len(data) // max_batch_size
    if len(data) % max_batch_size != 0:
        iterations += 1

    if not show_progress and desc is not None:
        print(desc)

    # Split the data into batches of max_batch_size and yield each batch
    for i in iter_with_progress(iterations, show_progress, desc):
        yield data[i * max_batch_size: (i + 1) * max_batch_size]


class ProgressBarTask:
    """
    Base class for progress bar tasks.
    """

    def __init__(self, _manager: enlighten.Manager, desc=None, leave=False):
        """
        Initialize a progress bar task.

        Args:
            _manager (enlighten.Manager): The manager object for progress bars.
            desc (str, optional): Description for the progress bar. Defaults to None.
            leave (bool, optional): Whether to leave the progress bar after completion. Defaults to False.
        """
        self._manager = _manager
        self._current_task = None
        self.desc = desc
        self.leave = leave

    def set_description(self, desc: str):
        """
        Set the description for the progress bar.

        Args:
            desc (str): Description for the progress bar.
        """
        self.desc = desc
        if self._current_task is not None:
            self._current_task.desc = desc

    def __iter__(self):
        """
        Iterate over the progress bar task.
        """
        raise Exception('not implemented yet')


class RangeProgressBar(ProgressBarTask):
    """
    Progress bar task for iterating over a range.
    """

    def __init__(self, _manager: enlighten.Manager, *args, desc=None, leave=False):
        """
        Initialize a range progress bar task.

        Args:
            _manager (enlighten.Manager): The manager object for progress bars.
            *args: Arguments for the range function.
            desc (str, optional): Description for the progress bar. Defaults to None.
            leave (bool, optional): Whether to leave the progress bar after completion. Defaults to False.
        """
        super().__init__(_manager, desc, leave)
        self.range = range(*args)

    def __iter__(self):
        """
        Iterate over the range progress bar task.
        """
        self._current_task = self._manager.counter(desc=self.desc, total=len(self.range), leave=self.leave)
        for x in self.range:
            yield x
            self._current_task.update()
        self._manager.remove(self._current_task)
        self.current_task = None


class IterProgressBar(ProgressBarTask):
    """
    Progress bar task for iterating over an iterable.
    """

    def __init__(self, _manager: enlighten.Manager, iterable: Iterable, total=None, desc=None, leave=False,
                 batch_size=None):
        """
        Initialize an iterable progress bar task.

        Args:
            _manager (enlighten.Manager): The manager object for progress bars.
            iterable (Iterable): The iterable object to iterate over.
            total (int, optional): The total number of elements in the iterable. Defaults to None.
            desc (str, optional): Description for the progress bar. Defaults to None.
            leave (bool, optional): Whether to leave the progress bar after completion. Defaults to False.
            batch_size (int, optional): The batch size for yielding elements. Defaults to None.
        """
        super().__init__(_manager, desc, leave)
        self.iterable = iterable
        self.batch_size = batch_size
        self.total = total
        if isinstance(iterable, Sized):
            self.total = len(iterable)

    def __iter__(self):
        """
        Iterate over the iterable progress bar task.
        """
        total = self.total
        if self.batch_size is not None and self.total is not None:
            total = self.total // self.batch_size
            if self.total % self.batch_size != 0:
                total += 1
        self._current_task = self._manager.counter(desc=self.desc, total=total, leave=self.leave)

        if self.batch_size is None:
            # Iterate over the iterable and update the progress bar.
            for x in self.iterable:
                yield x
                self._current_task.update()
        elif total is None or not hasattr(self.iterable, "__getitem__"):
            result = []
            # Iterate over the iterable in batches and update the progress bar.
            for x in self.iterable:
                result.append(x)
                if len(result) >= self.batch_size:
                    yield result
                    result = []
                    self._current_task.update()
            if len(result) > 0:
                yield result
                self._current_task.update()
        else:
            # Iterate over the iterable in batches using indexing and update the progress bar.
            for b in range(total):
                yield self.iterable[b * self.batch_size: (b + 1) * self.batch_size]
                self._current_task.update()

        self.set_description('[COMPLETED]')
        self._current_task.refresh()
        self._manager.remove(self._current_task)
        self.current_task = None


class RangeCounter:
    def __init__(self, _manager: enlighten.Manager, total, desc=None, leave=False):
        self._manager = _manager
        self.task = _manager.counter(total=total, desc=desc, leave=leave)
        self.count = 0
        self.total = total

    def update(self, incr: int = 1, desc=None):
        if desc is not None:
            self.task.desc = desc
        self.task.update(incr)
        self.count += incr
        if self.count == self.total:
            self.task.refresh()
            self._manager.remove(self.task)
            self.task = None


class ProgressBarManager:
    """
    Manager class for progress bars.
    """

    def __init__(self):
        """
        Initialize a progress bar manager.
        """
        self.manager = enlighten.get_manager()
        self.enabled = True

    def enable(self, mode=True):
        self.enabled = mode

    def disable(self):
        self.enable(False)

    def range(self, *args, desc=None, leave=False) -> ProgressBarTask:
        """
        Create a range progress bar task.

        Args:
            *args: Arguments for the range function.
            desc (str, optional): Description for the progress bar. Defaults to None.
            leave (bool, optional): Whether to leave the progress bar after completion. Defaults to False.

        Returns:
            ProgressBarTask: The range progress bar task.
        """
        return RangeProgressBar(self.manager, *args, desc=desc, leave=leave)

    def iter(self, iterable: Iterable, desc=None, leave=False, total=None, batch_size=None) -> ProgressBarTask:
        """
        Create an iterable progress bar task.

        Args:
            iterable (Iterable): The iterable object to iterate over.
            desc (str, optional): Description for the progress bar. Defaults to None.
            leave (bool, optional): Whether to leave the progress bar after completion. Defaults to False.
            total (int, optional): The total number of elements in the iterable. Defaults to None.
            batch_size (int, optional): The batch size for yielding elements. Defaults to None.

        Returns:
            ProgressBarTask: The iterable progress bar task.
        """
        return IterProgressBar(self.manager, iterable, total=total, desc=desc, leave=leave, batch_size=batch_size)

    def counter(self, total, desc=None, leave=False) -> RangeCounter:
        return RangeCounter(self.manager, total, desc, leave)

    def stop(self):
        """
        Stop the progress bar manager.
        """
        self.manager.stop()

    def __del__(self):
        """
        Clean up the progress bar manager.
        """
        self.stop()
