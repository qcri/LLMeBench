import unittest
from unittest.mock import patch

import openai

from llmebench import Benchmark
from llmebench.models import OpenAIModel


class TestAssetsForOpenAIPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the GPT model
        cls.assets = [
            asset for asset in all_assets if asset["config"]["model"] in [OpenAIModel]
        ]

    def test_openai_prompts(self):
        "Test if all assets using this model return data in an appropriate format for prompting"

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

    def test_openai_config(self):
        "Test if model config parameters passed as arguments are used"
        model = OpenAIModel(
            api_type="llmebench", api_key="secret-key", model_name="private-model"
        )

        self.assertEqual(openai.api_type, "llmebench")
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(model.model_params["model"], "private-model")

    def test_openai_config_azure(self):
        "Test if model config parameters passed as arguments are used (azure)"
        model = OpenAIModel(
            api_type="azure",
            api_key="secret-key",
            engine_name="private-model",
            api_version="v1",
            api_base="url.llmebench.org",
        )

        self.assertEqual(openai.api_type, "azure")
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(openai.api_version, "v1")
        self.assertEqual(model.model_params["engine"], "private-model")

    @patch.dict(
        "os.environ",
        {
            "AZURE_API_VERSION": "v1",
            "AZURE_API_URL": "url",
            "AZURE_API_KEY": "secret-key",
            "AZURE_ENGINE_NAME": "private-model",
        },
    )
    def test_openai_config_env_var_azure(self):
        "Test if model config parameters passed as environment variables are used (azure)"
        model = OpenAIModel()

        self.assertEqual(openai.api_type, "azure")
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(openai.api_version, "v1")
        self.assertEqual(model.model_params["engine"], "private-model")

    @patch.dict(
        "os.environ", {"OPENAI_API_KEY": "secret-key", "OPENAI_MODEL": "private-model"}
    )
    def test_openai_config_env_var_openai(self):
        "Test if model config parameters passed as environment variables are used (openai)"
        model = OpenAIModel()

        self.assertEqual(openai.api_type, "openai")
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(model.model_params["model"], "private-model")

    @patch.dict(
        "os.environ", {"OPENAI_API_KEY": "secret-key", "OPENAI_MODEL": "private-model"}
    )
    def test_openai_config_priority(self):
        "Test if model config parameters override environment variables"
        model = OpenAIModel(model_name="another-model")

        self.assertEqual(openai.api_type, "openai")
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(model.model_params["model"], "another-model")
