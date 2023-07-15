from typing import Sized, Iterable
import enlighten


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

        self._manager.remove(self._current_task)
        self.current_task = None


class ProgressBarManager:
    """
    Manager class for progress bars.
    """

    def __init__(self):
        """
        Initialize a progress bar manager.
        """
        self.manager = enlighten.get_manager()

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


if __name__ == '__main__':
    import time
    manager = ProgressBarManager()
    for i in manager.range(3, leave=True, desc='Process'):
        models_iter = manager.iter(['model1', 'model2', 'model3'])
        for m in models_iter:
            models_iter.set_description(f"Test {m}")
            k = []
            for j in manager.iter([f"t-{_k}" for _k in range(71)], desc="Training..", batch_size=10, total=98374):
                time.sleep(0.5)
                k.append(j)
            print(k)
    manager.stop()

