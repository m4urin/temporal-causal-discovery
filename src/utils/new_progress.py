import time
from collections.abc import Collection
from typing import Union

from rich.progress import Progress


class MyTask:
    def __init__(self, manager: Progress, iterable: Union[int, Collection], desc=None):
        self.manager = manager
        self.desc = "" if desc is None else desc
        if isinstance(iterable, int):
            self.task = manager.add_task(self.desc, total=iterable)
            self.iterable = range(iterable)
        elif isinstance(iterable, Collection):
            self.task = manager.add_task(self.desc, total=len(iterable))
            self.iterable = iterable
        else:
            raise Exception

    def __iter__(self):
        self.manager.reset(self.task, description=self.desc)
        for i in self.iterable:
            yield i
            self.manager.update(self.task, advance=1)

    def set_description(self, desc):
        self.manager.update(self.task, description=desc)


class ProgressBars:
    def __init__(self):
        self.manager = Progress(refresh_per_second=3)
        self.tasks = {}
        self.manager.start()

    def add_task(self, _id: str, iterable: Union[int, Collection], desc=None):
        if desc is None:
            desc = _id
        self.tasks[_id] = MyTask(self.manager, iterable, desc)

    def __del__(self):
        self.manager.stop()

    def __getitem__(self, item) -> MyTask:
        return self.tasks[item]

    def stop(self):
        self.manager.stop()


pbar = ProgressBars()
pbar.add_task('models', ['m1', 'm2', 'm3'])
pbar.add_task('eval', 3, "Hyper-parameter optimization..")
pbar.add_task('training', 70, "Training model..")


for m in pbar['models']:
    pbar['models'].set_description(f"Model {m}")
    for e in pbar['eval']:
        for t in pbar['training']:
            time.sleep(0.05)
            #print(m, e, t)

pbar.stop()
