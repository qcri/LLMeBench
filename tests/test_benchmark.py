import json
import sys
import types

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import MagicMock, patch

import llmebench

from llmebench import Benchmark
from llmebench.datasets.dataset_base import DatasetBase
from llmebench.models.model_base import ModelBase
from llmebench.tasks.task_base import TaskBase


class MockDataset(DatasetBase):
    @staticmethod
    def metadata():
        return {
            "splits": {
                "train": ["default_train_data1", "default_train_data2"],
                "dev": ["default_dev_data1", "default_dev_data2"],
                "test": ["default_test_data1", "default_test_data2"],
            }
        }

    @staticmethod
    def get_data_sample():
        return {"input": "input", "label": "label"}

    def load_data(self, data_path):
        return [{"input_id": i, **self.get_data_sample()} for i in data_path]


class MockDatasetWithMultiLevelSplits(DatasetBase):
    @staticmethod
    def metadata():
        return {
            "splits": {
                "ar": {
                    "dev": ["default_ar_dev_data1", "default_ar_dev_data2"],
                    "test": ["default_ar_test_data1", "default_ar_test_data2"],
                },
                "en": {
                    "dev": ["default_en_dev_data1", "default_en_dev_data2"],
                    "test": ["default_en_test_data1", "default_en_test_data2"],
                },
                "default": ["ar", "en"],
            }
        }

    @staticmethod
    def get_data_sample():
        return {"input": "input", "label": "label"}

    def load_data(self, data_path):
        return [{"input_id": i, **self.get_data_sample()} for i in data_path]


class MockModel(ModelBase):
    def prompt(self, processed_input):
        return processed_input

    def summarize_response(self, response):
        return response


class MockTask(TaskBase):
    def evaluate(self, true_labels, predicted_labels):
        return {"Accuracy": 1}


class MockAsset(object):
    @staticmethod
    def config():
        return {
            "dataset": MockDataset,
            "dataset_args": {"data_dir": ""},
            "task": MockTask,
            "model": MockModel,
        }

    @staticmethod
    def prompt(input_sample):
        return {"prompt": input_sample}

    @staticmethod
    def post_process(response):
        return response


class MockFewShotAsset(object):
    @staticmethod
    def config():
        return {
            "dataset": MockDataset,
            "dataset_args": {"data_dir": ""},
            "task": MockTask,
            "model": MockModel,
        }

    @staticmethod
    def prompt(input_sample, samples):
        return {"prompt": input_sample}

    @staticmethod
    def post_process(response):
        return response


class MockFailingAsset(MockAsset):
    def prompt(input_sample):
        raise Exception("Fail!")


class MockMultiConfigAsset(MockAsset):
    @staticmethod
    def config():
        return [
            {"name": "Subasset 1", "config": MockAsset.config()},
            {"name": "Subasset 2", "config": MockAsset.config()},
        ]


@patch("llmebench.utils.import_source_file", MagicMock(return_value=MockAsset))
class TestBenchmarkAssetFinder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets_dir = TemporaryDirectory()
        cls.assets_path = cls.assets_dir.name

        cls.assets = [
            "generic_zeroshot.py",
            "ar/unique_arabic_asset.py",
            "ar/sentiment/sentiment_zeroshot.py",
            "ar/sentiment/sentiment_fewshot.py",
            "en/sentiment/sentiment_zeroshot.py",
            "unique_prefix/task_1/asset_1.py",
            "unique_prefix/task_1/asset_2.py",
            "unique_prefix/task_1/asset_3.py",
            "unique_prefix/task_2/asset_1.py",
        ]

        non_assets = ["sometempfile.txt" "ar/arabic_runner.pyc"]

        for asset in cls.assets + non_assets:
            asset_path = Path(cls.assets_path) / asset
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.touch(exist_ok=True)

        cls.benchmark = Benchmark(benchmark_dir=cls.assets_path)

    @classmethod
    def tearDownClass(cls):
        cls.assets_dir.cleanup()

    def test_default(self):
        "Test if all assets are found when benchmark is run"

        self.assertEqual(len(self.benchmark.find_assets()), len(self.assets))

    def test_filename(self):
        "Test if an asset is found by exact file name"

        assets = self.benchmark.find_assets("unique_arabic_asset.py")

        self.assertEqual(len(assets), 1)
        self.assertIn("unique_arabic_asset", assets[0]["name"])

    def test_filename_root(self):
        "Test if an asset is found by exact file name when in the root of the assets folder"

        assets = self.benchmark.find_assets("generic_zeroshot.py")

        self.assertEqual(len(assets), 1)
        self.assertIn("generic_zeroshot", assets[0]["name"])

    def test_wildcard(self):
        "Test if assets are found with wildcard search"

        assets = self.benchmark.find_assets("*zeroshot*")

        self.assertEqual(len(assets), 3)
        for asset in assets:
            self.assertIn("zeroshot", asset["name"])

    def test_full_absolute_path(self):
        "Test if assets are found with their full absolute path"

        assets = self.benchmark.find_assets(
            self.assets_path + "/ar/sentiment/sentiment_zeroshot.py"
        )

        self.assertEqual(len(assets), 1)
        self.assertIn("sentiment_zeroshot", assets[0]["name"])

    def test_full_relative_path(self):
        "Test if assets are found with their full relative path"

        assets = self.benchmark.find_assets("ar/sentiment/sentiment_zeroshot.py")

        self.assertEqual(len(assets), 1)
        self.assertIn("sentiment_zeroshot", assets[0]["name"])

    def test_partial_path(self):
        "Test if assets are found with their partial paths"

        assets = self.benchmark.find_assets("unique_prefix/*")

        self.assertEqual(len(assets), 4)
        for asset in assets:
            self.assertIn("unique_prefix/", asset["name"])


class TestBenchmarkRunner(unittest.TestCase):
    def setUp(self):
        self.benchmark_dir = TemporaryDirectory()
        self.results_dir = TemporaryDirectory()

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_no_asset_run(self, asset_finder_mock):
        "Run benchmark with no assets"
        asset_finder_mock.return_value = []

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 0)

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_single_asset_run(self, asset_finder_mock):
        "Run benchmark with one asset"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": MockAsset.config(),
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_single_failing_asset_run(self, asset_finder_mock):
        "Run benchmark with one failing asset"
        asset_finder_mock.return_value = [
            {
                "name": "MockFailingAsset",
                "config": MockFailingAsset.config(),
                "module": MockFailingAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 0)

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_multiple_assets(self, asset_finder_mock):
        "Run benchmark with multiple assets"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": MockAsset.config(),
                "module": MockAsset,
            },
            {
                "name": "MockAsset 2",
                "config": MockAsset.config(),
                "module": MockAsset,
            },
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 2)

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_multiple_assets_with_failure(self, asset_finder_mock):
        "Run benchmark with multiple assets and failing assets"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": MockAsset.config(),
                "module": MockAsset,
            },
            {
                "name": "MockFailingAsset 1",
                "config": MockFailingAsset.config(),
                "module": MockFailingAsset,
            },
            {
                "name": "MockAsset 2",
                "config": MockAsset.config(),
                "module": MockAsset,
            },
            {
                "name": "MockFailingAsset 2",
                "config": MockFailingAsset.config(),
                "module": MockFailingAsset,
            },
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 2)

    @patch("llmebench.utils.import_source_file")
    def test_multi_config_asset(self, asset_importer_mock):
        "Run benchmark with multiconfig asset"

        # Create dummy asset file
        (Path(self.benchmark_dir.name) / "sample.py").touch(exist_ok=True)

        asset_importer_mock.return_value = MockMultiConfigAsset

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                config = MockMultiConfigAsset.config()
                self.assertEqual(len(results), len(config))

                for subconfig in config:
                    self.assertIn(f"sample/{subconfig['name']}", results)

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_default_splits(self, asset_finder_mock):
        "Run benchmark with an asset and its default splits"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": MockAsset.config(),
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

            with open(Path(self.results_dir.name) / "MockAsset 1" / "0.json") as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_test_data1")

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_selected_split(self, asset_finder_mock):
        "Run benchmark with an asset and a selected split"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {**MockAsset.config(), "general_args": {"test_split": "dev"}},
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

            with open(Path(self.results_dir.name) / "MockAsset 1" / "0.json") as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_dev_data1")

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_selected_splits(self, asset_finder_mock):
        "Run benchmark with an asset and a selected splits"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {
                    **MockAsset.config(),
                    "general_args": {"test_split": ["dev", "test"]},
                },
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 2)

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "dev" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_dev_data1")

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "test" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_test_data1")

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_custom_splits(self, asset_finder_mock):
        "Run benchmark with an asset and a custom split"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {
                    **MockAsset.config(),
                    "general_args": {
                        "custom_test_split": ["custom_data_1", "custom_data_2"]
                    },
                },
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

            with open(Path(self.results_dir.name) / "MockAsset 1" / "0.json") as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "custom_data_1")

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_default_splits_multilevel(self, asset_finder_mock):
        "Run benchmark with an asset (containing multi-level splits) and its default splits"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {
                    **MockAsset.config(),
                    "dataset": MockDatasetWithMultiLevelSplits,
                },
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 2)

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "ar" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(
                    cache_obj["input"]["input_id"], "default_ar_test_data1"
                )

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "en" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(
                    cache_obj["input"]["input_id"], "default_en_test_data1"
                )

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_selected_split_multilevel(self, asset_finder_mock):
        "Run benchmark with an asset (containing multi-level splits) and selected split"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {
                    **MockAsset.config(),
                    "dataset": MockDatasetWithMultiLevelSplits,
                    "general_args": {"test_split": ["ar/dev"]},
                },
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

            with open(Path(self.results_dir.name) / "MockAsset 1" / "0.json") as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_ar_dev_data1")

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_selected_split_multilevel_shorthand(self, asset_finder_mock):
        "Run benchmark with an asset (containing multi-level splits) and selected split (using shorthand)"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {
                    **MockAsset.config(),
                    "dataset": MockDatasetWithMultiLevelSplits,
                    "general_args": {"test_split": ["ar"]},
                },
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

            with open(Path(self.results_dir.name) / "MockAsset 1" / "0.json") as fp:
                cache_obj = json.load(fp)
                self.assertEqual(
                    cache_obj["input"]["input_id"], "default_ar_test_data1"
                )

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_asset_with_selected_splits_multilevel(self, asset_finder_mock):
        "Run benchmark with an asset (containing multi-level splits) and selected splits"
        asset_finder_mock.return_value = [
            {
                "name": "MockAsset 1",
                "config": {
                    **MockAsset.config(),
                    "dataset": MockDatasetWithMultiLevelSplits,
                    "general_args": {"test_split": ["ar/dev", "en/dev", "en/test"]},
                },
                "module": MockAsset,
            }
        ]

        testargs = ["llmebench", self.benchmark_dir.name, self.results_dir.name]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 3)

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "ar" / "dev" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_ar_dev_data1")

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "en" / "dev" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(cache_obj["input"]["input_id"], "default_en_dev_data1")

            with open(
                Path(self.results_dir.name) / "MockAsset 1" / "en" / "test" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                self.assertEqual(
                    cache_obj["input"]["input_id"], "default_en_test_data1"
                )

    @patch("llmebench.benchmark.Benchmark.find_assets")
    def test_fewshot_asset_with_default_splits(self, asset_finder_mock):
        "Run benchmark with an asset and its default splits"
        asset_finder_mock.return_value = [
            {
                "name": "MockFewShotAsset 1",
                "config": MockFewShotAsset.config(),
                "module": MockFewShotAsset,
            }
        ]

        testargs = [
            "llmebench",
            "--n_shots",
            "3",
            self.benchmark_dir.name,
            self.results_dir.name,
        ]
        with patch.object(sys, "argv", testargs):
            llmebench.benchmark.main()

            with open(Path(self.results_dir.name) / "all_results.json") as fp:
                results = json.load(fp)
                self.assertEqual(len(results), 1)

            print(list((Path(self.results_dir.name) / "MockFewShotAsset 1").iterdir()))

            with open(
                Path(self.results_dir.name) / "MockFewShotAsset 1" / "3_shot" / "0.json"
            ) as fp:
                cache_obj = json.load(fp)
                for fse in cache_obj["few_shot_examples"]:
                    self.assertIn("train", fse["input_id"])
