import unittest
import os
import shutil

from src.utils.io import *


class TestFunctions(unittest.TestCase):

    def setUp(self):
        os.makedirs('test')

    def tearDown(self):
        shutil.rmtree('test')

    def test_join_path(self):
        self.assertEqual(join_path("test", "folder", "file.txt"), os.path.abspath("test/folder/file.txt"))
        self.assertEqual(join_path("test", "new_folder"), os.path.abspath("test/new_folder"))

    def test_load_json_file_save_json_file(self):
        data = {"key": "value", "number": 42}
        filepath = "test/test.json"
        save_json_file(filepath, data)
        loaded_data = load_json_file(filepath)
        self.assertEqual(loaded_data, data)
        os.remove(filepath)

    def test_load_bz2_file_save_bz2_file(self):
        data = {"key": "value", "number": 42}
        filepath = "test/test.bz2"
        save_bz2_file(filepath, data)
        loaded_data = load_bz2_file(filepath)
        self.assertEqual(loaded_data, data)
        os.remove(filepath)

    def test_load_pickled_object_save_object_pickled(self):
        data = {"key": "value", "number": 42}
        filepath = "test/test.pickle"
        save_object_pickled(filepath, data)
        loaded_data = load_pickled_object(filepath)
        self.assertEqual(loaded_data, data)
        os.remove(filepath)


if __name__ == '__main__':
    unittest.main()
