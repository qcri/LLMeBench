import unittest
from unittest.mock import patch

import openai

from llmebench import Benchmark
from llmebench.models import FastChatModel

from tests.models.test_OpenAIModel import TestAssetsForOpenAIPrompts


class TestAssetsForFastChatPrompts(TestAssetsForOpenAIPrompts):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the GPT model
        cls.assets = [
            asset for asset in all_assets if asset["config"]["model"] in [FastChatModel]
        ]

    def test_fastchat_prompts(self):
        "Test if all assets using this model return data in an appropriate format for prompting"

        self.test_openai_prompts()


class TestFastChatConfig(unittest.TestCase):
    def test_fastchat_config(self):
        "Test if model config parameters passed as arguments are used"
        model = FastChatModel(
            api_base="llmebench.qcri.org",
            api_key="secret-key",
            model_name="private-model",
        )

        self.assertEqual(openai.api_type, "openai")
        self.assertEqual(
            model.client.base_url.raw_path.decode("utf-8"), "llmebench.qcri.org/"
        )
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(model.model_params["model"], "private-model")

    @patch.dict(
        "os.environ",
        {
            "FASTCHAT_API_BASE": "llmebench.qcri.org",
            "FASTCHAT_API_KEY": "secret-key",
            "FASTCHAT_MODEL": "private-model",
        },
    )
    def test_fastchat_config_env_var(self):
        "Test if model config parameters passed as environment variables are used"
        model = FastChatModel()

        self.assertEqual(openai.api_type, "openai")
        self.assertEqual(
            model.client.base_url.raw_path.decode("utf-8"), "llmebench.qcri.org/"
        )
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(model.model_params["model"], "private-model")

    @patch.dict(
        "os.environ",
        {
            "FASTCHAT_API_BASE": "llmebench.qcri.org",
            "FASTCHAT_API_KEY": "secret-key",
            "FASTCHAT_MODEL": "private-model",
        },
    )
    def test_fastchat_config_priority(self):
        "Test if model config parameters override environment variables"
        model = FastChatModel(model_name="another-model")

        self.assertEqual(openai.api_type, "openai")
        self.assertEqual(
            model.client.base_url.raw_path.decode("utf-8"), "llmebench.qcri.org/"
        )
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(model.model_params["model"], "another-model")
