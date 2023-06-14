import types

import unittest
from unittest.mock import patch

from arabic_llm_benchmark import Benchmark


class TestBenchmarkAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        benchmark = Benchmark(benchmark_dir="assets")

        cls.assets = benchmark.find_runs()

    def test_required_functions(self):
        "Test if all assets have required functions"

        for asset_idx, asset in enumerate(self.assets):
            with self.subTest(msg=asset["name"], i=asset_idx):
                self.assertIsInstance(asset["module"].config, types.FunctionType)
                self.assertIsInstance(asset["module"].prompt, types.FunctionType)
                self.assertIsInstance(asset["module"].post_process, types.FunctionType)

    @patch("os.environ")
    def test_config_format(self, os_env_mock):
        "Test if all configs are well defined"

        # Handle environment variables required at runtime
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["module"].config()

                self.assertIn("dataset", config)
                self.assertIn("dataset_args", config)
                self.assertIn("task", config)
                self.assertIn("task_args", config)
                self.assertIn("model", config)
                self.assertIn("model_args", config)
                self.assertIn("general_args", config)

                if "fewshot" in config["general_args"]:
                    self.assertIn("train_data_path", config["general_args"]["fewshot"])
                    self.assertIn("n_shots", config["general_args"]["fewshot"])
                    self.assertIsInstance(
                        config["general_args"]["fewshot"]["n_shots"], int
                    )
