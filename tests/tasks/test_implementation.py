import inspect
import unittest

from pathlib import Path

import llmebench.tasks as tasks

from tests.utils import base_class_constructor_checker


class TestTaskImplementation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented models
        framework_dir = Path("llmebench")
        cls.tasks = set([m[1] for m in inspect.getmembers(tasks, inspect.isclass)])

    def test_base_constructor(self):
        "Test if all tasks also call the base class constructor"

        for task in self.tasks:
            with self.subTest(msg=task.__name__):
                base_class_constructor_checker(task, self)
