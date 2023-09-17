import inspect
import unittest

from pathlib import Path

import llmebench.datasets as datasets

from tests.utils import base_class_constructor_checker


class TestDatasetImplementation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented datasets
        framework_dir = Path("llmebench")
        cls.datasets = set(
            [m[1] for m in inspect.getmembers(datasets, inspect.isclass)]
        )

    def test_base_constructor(self):
        "Test if all datasets also call the base class constructor"

        for dataset in self.datasets:
            with self.subTest(msg=dataset.__name__):
                base_class_constructor_checker(dataset, self)

    def test_metadata_static(self):
        "Test if all datasets mark metadata method as static"

        for dataset in self.datasets:
            with self.subTest(msg=dataset.__name__):
                method_impl = dataset.__dict__.get("metadata")
                if method_impl:
                    self.assertIsInstance(method_impl, staticmethod)

    def test_get_data_sample_static(self):
        "Test if all datasets mark get_data_sample method as static"

        for dataset in self.datasets:
            with self.subTest(msg=dataset.__name__):
                method_impl = dataset.__dict__.get("get_data_sample")
                if method_impl:
                    self.assertIsInstance(method_impl, staticmethod)
