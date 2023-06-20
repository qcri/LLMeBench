import types

import unittest
from unittest.mock import patch

from arabic_llm_benchmark import Benchmark


class TestBenchmarkAssets(unittest.TestCase):
    @classmethod
    @patch("os.environ")
    def setUpClass(cls, os_env_mock):
        # Handle environment variables required at runtime
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        benchmark = Benchmark(benchmark_dir="assets")

        cls.assets = benchmark.find_assets()

    def test_required_functions(self):
        "Test if all assets have required functions"

        for asset_idx, asset in enumerate(self.assets):
            with self.subTest(msg=asset["name"], i=asset_idx):
                self.assertIsInstance(asset["module"].config, types.FunctionType)
                self.assertIsInstance(asset["module"].prompt, types.FunctionType)
                self.assertIsInstance(asset["module"].post_process, types.FunctionType)

    def validate_single_config(self, config):
        self.assertIn("dataset", config)
        self.assertIn("dataset_args", config)
        self.assertIn("task", config)
        self.assertIn("task_args", config)
        self.assertIn("model", config)
        self.assertIn("model_args", config)
        self.assertIn("general_args", config)

        if "fewshot" in config["general_args"]:
            self.assertIn("train_data_path", config["general_args"]["fewshot"])

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
