import os

import unittest

from unittest.mock import patch

from arabic_llm_benchmark import Benchmark
from arabic_llm_benchmark.models import BLOOMPetalModel


class TestAssetsForBLOOMPetalPrompts(unittest.TestCase):
    @classmethod
    @patch("os.environ")
    def setUpClass(cls, os_env_mock):
        # Handle environment variables required at runtime
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_runs()

        # Filter out assets not using the BLOOMPetal model
        cls.assets = [
            asset
            for asset in all_assets
            if asset["module"].config()["model"] in [BLOOMPetalModel]
        ]

    @patch("os.environ")
    def test_gpt_prompts(self, os_env_mock):
        "Test if all assets using this model return data in an appropriate format for prompting"
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        n_shots = 3  # Sample for few shot prompts

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["module"].config()
                dataset = config["dataset"](**config["dataset_args"])
                data_sample = dataset.get_data_sample()
                if "fewshot" in config["general_args"]:
                    prompt = asset["module"].prompt(
                        data_sample["input"],
                        [data_sample for _ in range(n_shots)],
                    )
                else:
                    prompt = asset["module"].prompt(data_sample["input"])

                self.assertIsInstance(prompt, dict)
                self.assertIn("prompt", prompt)
                self.assertIsInstance(prompt["prompt"], str)
