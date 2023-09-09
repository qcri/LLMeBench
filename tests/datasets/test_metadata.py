import inspect
import unittest

from pathlib import Path

import llmebench.datasets as datasets

from langcodes import tag_is_valid


class TestDatasetMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Search for all implemented datasets
        framework_dir = Path("llmebench")
        cls.datasets = set(
            [m[1] for m in inspect.getmembers(datasets, inspect.isclass)]
        )

    def test_dataset_metadata(self):
        "Test if all datasets export the required metadata"

        for dataset in self.datasets:
            with self.subTest(msg=dataset.__name__):
                self.assertIsInstance(dataset.metadata(), dict)
                self.assertIn("citation", dataset.metadata())
                self.assertIsInstance(dataset.metadata()["citation"], str)
                self.assertIn("language", dataset.metadata())
                self.assertIsInstance(dataset.metadata()["language"], (str, list))

                languages = dataset.metadata()["language"]
                if isinstance(languages, str):
                    languages = [languages]

                for language in languages:
                    self.assertTrue(
                        language == "multilingual" or tag_is_valid(language),
                        f"{language} is not a valid language",
                    )
