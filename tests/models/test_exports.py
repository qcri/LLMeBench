import inspect
import unittest

from glob import glob
from pathlib import Path

import llmebench.models as models
from llmebench import utils
from llmebench.models.model_base import ModelBase


class TestDatasetExports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented models
        framework_dir = Path("llmebench")
        cls.implemented_models = [
            model_path
            for model_path in glob(str(framework_dir / "models" / "*.py"))
            if "__init__" not in model_path and "model_base" not in model_path
        ]
        cls.exported_models = set(
            [m[1].__name__ for m in inspect.getmembers(models, inspect.isclass)]
        )

    def test_model_exports(self):
        "Test if all implemented models are exported in __init__.py"

        for model_path in self.implemented_models:
            with self.subTest(msg=model_path):
                implemented_module = utils.import_source_file(
                    Path(model_path), "test_module"
                )
                implemented_class = [
                    c
                    for c in inspect.getmembers(implemented_module, inspect.isclass)
                    if issubclass(c[1], ModelBase) and c[1] != ModelBase
                ][0]
                implemented_model = implemented_class[1].__name__

                self.assertIn(
                    implemented_model,
                    self.exported_models,
                    msg=f"{implemented_model} not exported in models/__init__.py",
                )
