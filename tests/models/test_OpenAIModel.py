import unittest
from unittest.mock import patch

import openai

from llmebench import Benchmark
from llmebench.models import OpenAIModel

from llmebench.utils import is_fewshot_asset


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
                            elif elem["type"] == "input_audio":
                                self.assertIn("input_audio", elem)
                                self.assertIsInstance(elem["input_audio"], dict)

                                self.assertIn("data", elem["input_audio"])
                                self.assertIsInstance(elem["input_audio"]["data"], str)

                                self.assertIn("format", elem["input_audio"])
                                self.assertEqual(elem["input_audio"]["format"], "wav")


class TestOpenAIConfig(unittest.TestCase):
    def test_openai_config(self):
        "Test if model config parameters passed as arguments are used"
        model = OpenAIModel(api_key="secret-key", model_name="private-model")

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
        self.assertEqual(model.model_params["model"], "private-model")

    @patch.dict(
        "os.environ",
        {
            "AZURE_API_VERSION": "v1",
            "AZURE_API_URL": "url",
            "AZURE_API_KEY": "secret-key",
            "AZURE_ENGINE_NAME": "private-model",
            "OPENAI_API_BASE": "",
            "OPENAI_API_KEY": "",
            "OPENAI_MODEL": "",
        },
    )
    def test_openai_config_env_var_azure(self):
        "Test if model config parameters passed as environment variables are used (azure)"
        model = OpenAIModel()

        self.assertEqual(openai.api_type, "azure")
        self.assertEqual(openai.api_key, "secret-key")
        self.assertEqual(openai.api_version, "v1")
        self.assertEqual(model.model_params["model"], "private-model")

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
