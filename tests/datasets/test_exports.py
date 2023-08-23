import inspect
import unittest

from glob import glob
from pathlib import Path

import llmebench.datasets as datasets
from llmebench import utils
from llmebench.datasets.dataset_base import DatasetBase


class TestDatasetExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented datasets
        framework_dir = Path("llmebench")
        cls.implemented_datasets = [
            dataset_path
            for dataset_path in glob(str(framework_dir / "datasets" / "*.py"))
            if "__init__" not in dataset_path and "dataset_base" not in dataset_path
        ]
        cls.exported_datasets = set(
            [m[1].__name__ for m in inspect.getmembers(datasets, inspect.isclass)]
        )

    def test_dataset_exports(self):
        "Test if all implemented datasets are exported in __init__.py"

        for dataset_path in self.implemented_datasets:
            with self.subTest(msg=dataset_path):
                implemented_module = utils.import_source_file(
                    Path(dataset_path), "test_module"
                )
                implemented_class = [
                    c
                    for c in inspect.getmembers(implemented_module, inspect.isclass)
                    if issubclass(c[1], DatasetBase) and c[1] != DatasetBase
                ][0]

                # Base classes do not need to be exported
                if inspect.isabstract(implemented_class[1]):
                    continue

                implemented_dataset = implemented_class[1].__name__

                self.assertIn(
                    implemented_dataset,
                    self.exported_datasets,
                    msg=f"{implemented_dataset} not exported in datasets/__init__.py",
                )
