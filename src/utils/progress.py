from datetime import datetime, timedelta
from typing import Generator, Iterable

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

    generator = iter_with_progress(iterations, desc) if show_progress else range(iterations)
    # Split the data into batches of max_batch_size and yield each batch
    for i in generator:
        yield data[i * max_batch_size: (i + 1) * max_batch_size]


def test():
    """
    Tests the functionality of the provided code.
    """
    import time
    global USE_CONSOLE

    # Test `ProgressWriter` class
    pw = ProgressWriter(total=10, desc="ProgressWriter Test")
    pw.update()
    time.sleep(1)  # Wait for 1 second to see the progress update
    pw.update()
    print()

    pw.reset(total=20, desc="ProgressWriter Reset Test")
    for _ in range(10):
        pw.update()
        time.sleep(0.2)  # Wait for 1 second to see the progress update
    print()

    # Test `tqdm_print` function
    data = list(range(10))
    for _ in tqdm_print(data, "test tqdm_print"):
        time.sleep(0.2)  # Wait for 0.1 second to see the progress update
    print()

    # Test `trange_print` function
    for _ in trange_print(total=10, desc="test trange_print"):
        time.sleep(0.2)  # Wait for 0.1 second to see the progress update
    print()

    # Test `iter_with_progress` function with 'tqdm' mode
    USE_CONSOLE = False
    for _ in iter_with_progress(total=10, desc="test iter_with_progress (pc)"):
        time.sleep(0.2)  # Wait for 0.1 second to see the progress update
    print()

    # Test `iter_with_progress` function with 'console' mode
    USE_CONSOLE = True
    for _ in iter_with_progress(total=10, desc="test iter_with_progress (gpulab)"):
        time.sleep(0.2)  # Wait for 0.1 second to see the progress update

    print("\nTest passed!")

