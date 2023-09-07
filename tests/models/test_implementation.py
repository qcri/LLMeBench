import inspect
import unittest

from pathlib import Path

import llmebench.models as models

from tests.utils import base_class_constructor_checker


class TestModelImplementation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented models
        framework_dir = Path("llmebench")
        cls.models = set([m[1] for m in inspect.getmembers(models, inspect.isclass)])

    def test_base_constructor(self):
        "Test if all models also call the base class constructor"

        for model in self.models:
            with self.subTest(msg=model.__name__):
                base_class_constructor_checker(model, self)
