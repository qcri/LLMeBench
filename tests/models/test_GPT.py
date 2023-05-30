import os

import unittest

from unittest import mock

from arabic_llm_benchmark import Benchmark
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel


class TestBenchmarkAssets(unittest.TestCase):
    @classmethod
    @mock.patch.dict(
        os.environ,
        {"AZURE_API_URL": "test_url", "AZURE_API_KEY": "test_key"},
        clear=True,
    )
    def setUpClass(cls):
        benchmark = Benchmark(benchmark_dir="assets")

        all_assets = benchmark.find_runs()

        cls.assets = [
            asset
            for asset in all_assets
            if asset["module"].config()["model"] in [GPTModel, RandomGPTModel]
        ]
        print(cls.assets)

    def test_gpt_prompts(self):
        "Test if all assets have required functions"

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                prompt = asset["module"].prompt("some_input")

                self.assertIsInstance(prompt, dict)
                self.assertIn("system_message", prompt)
                self.assertIsInstance(prompt["system_message"], str)
                self.assertIn("messages", prompt)
                self.assertIsInstance(prompt["messages"], list)

                for message in prompt["messages"]:
                    self.assertIsInstance(message, dict)
                    self.assertIn("sender", message)
                    self.assertIsInstance(message["sender"], str)
                    self.assertIn("text", message)
                    self.assertIsInstance(message["text"], str)
