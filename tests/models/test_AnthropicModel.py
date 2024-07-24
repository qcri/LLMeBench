import unittest
from unittest.mock import patch

from llmebench import Benchmark
from llmebench.models import AnthropicModel

from llmebench.utils import is_fewshot_asset


class TestAssetsForAnthropicPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the Petals model
        cls.assets = [
            asset
            for asset in all_assets
            if asset["config"]["model"] in [AnthropicModel]
        ]

    def test_anthropic_prompts(self):
        "Test if all assets using this model return data in an appropriate format for prompting"
        # self.test_openai_prompts()
        n_shots = 3  # Sample for few shot prompts

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]
                dataset_args = config.get("dataset_args", {})
                dataset_args["data_dir"] = ""
                dataset = config["dataset"](**dataset_args)
                data_sample = dataset.get_data_sample()
                if is_fewshot_asset(config, asset["module"].prompt):
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
                    self.assertIsInstance(message["content"], (str, list))

                    # Multi-modal input
                    if isinstance(message["content"], list):
                        for elem in message["content"]:
                            self.assertIsInstance(elem, dict)
                            self.assertIn("type", elem)

                            if elem["type"] == "text":
                                self.assertIn("text", elem)
                                self.assertIsInstance(elem["text"], str)
                            elif elem["type"] == "image_url":
                                self.assertIn("image_url", elem)
                                self.assertIsInstance(elem["image_url"], dict)
                                self.assertIn("url", elem["image_url"])
                                self.assertIsInstance(elem["image_url"]["url"], str)


class TestAnthropicConfig(unittest.TestCase):
    def test_anthropic_config(self):
        "Test if model config parameters passed as arguments are used"
        model = AnthropicModel(api_key="secret-key", model_name="private-model")
        self.assertEqual(model.api_key, "secret-key")

    @patch.dict(
        "os.environ",
        {
            "ANTHROPIC_API_KEY": "secret-key",
            "ANTHROPIC_MODEL": "model",
        },
    )
    def test_anthropic_config_env_var(self):
        "Test if model config parameters passed as environment variables are used"
        model = AnthropicModel(api_key="secret-key", model_name="private-model")
        self.assertEqual(model.api_key, "secret-key")

    @patch.dict(
        "os.environ",
        {
            "ANTHROPIC_API_KEY": "secret-key",
            "ANTHROPIC_MODEL": "model",
        },
    )
    def test_anthropic_config_priority(self):
        "Test if model config parameters passed as environment variables are used"
        model = AnthropicModel(api_key="secret-key", model_name="private-model")

        self.assertEqual(model.api_key, "secret-key")
