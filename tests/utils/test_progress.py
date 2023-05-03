import time
import unittest

from src.utils.progress import *


class TestProgressWriter(unittest.TestCase):
    def test_progress_writer(self):
        pw = ProgressWriter(total=10, desc="ProgressWriter Test")
        pw.update()
        time.sleep(1)
        pw.update()

        pw.reset(total=20, desc="ProgressWriter Reset Test")
        for _ in range(10):
            pw.update()
            time.sleep(0.1)

    def test_tqdm_print(self):
        data = list(range(10))
        for _ in tqdm_print(data, "test tqdm_print"):
            time.sleep(0.1)

    def test_trange_print(self):
        for _ in trange_print(total=10, desc="test trange_print"):
            time.sleep(0.1)

    def test_iter_with_progress(self):
        use_console(False)
        for _ in iter_with_progress(total=10, desc="test iter_with_progress (pc)"):
            time.sleep(0.1)

        use_console(True)
        for _ in iter_with_progress(total=10, desc="test iter_with_progress (gpulab)"):
            time.sleep(0.1)


if __name__ == '__main__':
    unittest.main()
