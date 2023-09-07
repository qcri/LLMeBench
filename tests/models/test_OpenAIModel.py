import os

import unittest

from unittest.mock import patch

from llmebench import Benchmark
from llmebench.models import OpenAIModel


class TestAssetsForGPTChatCompletionPrompts(unittest.TestCase):
    @classmethod
    @patch("os.environ")
    def setUpClass(cls, os_env_mock):
        # Handle environment variables required at runtime
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the GPT model
        cls.assets = [
            asset for asset in all_assets if asset["config"]["model"] in [OpenAIModel]
        ]

    @patch("os.environ")
    def test_gpt_prompts(self, os_env_mock):
        "Test if all assets using this model return data in an appropriate format for prompting"
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        n_shots = 3  # Sample for few shot prompts

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]
                dataset = config["dataset"](**config["dataset_args"])
                data_sample = dataset.get_data_sample()
                if "fewshot" in config["general_args"]:
                    prompt = asset["module"].prompt(
                        data_sample["input"],
                        [data_sample for _ in range(n_shots)],
                    )
                else:
                    prompt = asset["module"].prompt(data_sample["input"])

                self.assertIsInstance(prompt, list)

                for message in prompt:
                    self.assertIsInstance(message, dict)
                    self.assertIn("role", message)
                    self.assertIsInstance(message["role"], str)
                    self.assertIn("content", message)
                    self.assertIsInstance(message["content"], str)
