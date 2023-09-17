import inspect
import unittest

from pathlib import Path

import llmebench.datasets as datasets

from langcodes import tag_is_valid
from llmebench.tasks import TaskType


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
                metadata = dataset.metadata()

                self.assertIsInstance(metadata, dict)
                self.assertIn("citation", metadata)
                self.assertIsInstance(metadata["citation"], str)
                self.assertIn("language", metadata)
                self.assertIsInstance(metadata["language"], (str, list))

                languages = metadata["language"]
                if isinstance(languages, str):
                    languages = [languages]

                for language in languages:
                    self.assertTrue(
                        language == "multilingual" or tag_is_valid(language),
                        f"{language} is not a valid language",
                    )

                self.assertIn("splits", metadata)
                for split_name in metadata["splits"]:
                    self.assertFalse(
                        "/" in split_name, "Split names cannot contain '/'"
                    )
                    if isinstance(metadata["splits"][split_name], dict):
                        for sub_split_name in metadata["splits"][split_name]:
                            self.assertFalse(
                                "/" in split_name, "Split names cannot contain '/'"
                            )

                self.assertIn("task_type", metadata)
                self.assertIsInstance(metadata["task_type"], TaskType)

                if metadata["task_type"] in [
                    TaskType.Classification,
                    TaskType.SequenceLabeling,
                    TaskType.MultiLabelClassification,
                ]:
                    self.assertIn("class_labels", metadata)
                    self.assertIsInstance(metadata["class_labels"], list)
                elif metadata["task_type"] == TaskType.Regression:
                    self.assertIn("score_range", metadata)
                    self.assertIsInstance(metadata["score_range"], tuple)
