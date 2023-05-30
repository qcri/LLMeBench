import os

import unittest
from collections import defaultdict
from unittest import mock

from arabic_llm_benchmark import Benchmark, utils


class TestBenchmarkAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        benchmark = Benchmark(benchmark_dir="assets")

        cls.assets = benchmark.find_runs()

    def test_required_functions(self):
        "Test if all assets have required functions"

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                asset["module"].config
                asset["module"].prompt
                asset["module"].post_process

    @mock.patch.dict(
        os.environ,
        {"AZURE_API_URL": "test_url", "AZURE_API_KEY": "test_key"},
        clear=True,
    )
    def test_config_format(self):
        "Test if all configs are well defined"

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
