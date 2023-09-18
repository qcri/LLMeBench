import types

import unittest

from llmebench import Benchmark


class TestBenchmarkAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        benchmark = Benchmark(benchmark_dir="assets")

        cls.assets = benchmark.find_assets()

    def test_required_functions(self):
        "Test if all assets have required functions"

        for asset_idx, asset in enumerate(self.assets):
            with self.subTest(msg=asset["name"], i=asset_idx):
                self.assertIsInstance(asset["module"].metadata, types.FunctionType)
                self.assertIsInstance(asset["module"].config, types.FunctionType)
                self.assertIsInstance(asset["module"].prompt, types.FunctionType)
                self.assertIsInstance(asset["module"].post_process, types.FunctionType)

    def validate_single_config(self, config):
        self.assertIn("dataset", config)

        if "dataset_args" in config:
            self.assertIsInstance(config["dataset_args"], dict)
        self.assertIn("task", config)
        if "task_args" in config:
            self.assertIsInstance(config["task_args"], dict)
        self.assertIn("model", config)
        if "model_args" in config:
            self.assertIsInstance(config["model_args"], dict)

        if "general_args" in config:
            self.assertIsInstance(config["general_args"], dict)

    def test_config_format(self):
        "Test if all configs are well defined"

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]

                self.assertIsInstance(config, (dict, list))

                if isinstance(config, dict):
                    self.validate_single_config(config)
                else:
                    for subconfig in config:
                        self.assertIn("name", subconfig)
                        self.assertIsInstance(subconfig["name"], str)
                        self.assertIn("config", subconfig)
                        self.assertIsInstance(subconfig["config"], dict)
                        self.validate_single_config(subconfig["config"])

    def test_metadata_format(self):
        "Test if metadata is well defined"

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                metadata = asset["module"].metadata()

                self.assertIsInstance(metadata, dict)

                self.assertIn("author", metadata)
                self.assertIsInstance(metadata["author"], str)
                self.assertIn("model", metadata)
                self.assertIsInstance(metadata["model"], str)
                self.assertIn("description", metadata)
                self.assertIsInstance(metadata["description"], str)
                self.assertIsInstance(metadata.get("scores", {}), dict)
