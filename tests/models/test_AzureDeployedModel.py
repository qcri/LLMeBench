import unittest
from unittest.mock import patch

from llmebench import Benchmark
from llmebench.models import AzureModel

from llmebench.utils import is_fewshot_asset


class TestAssetsForAzureDepModelPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the Petals model
        cls.assets = [
            asset for asset in all_assets if asset["config"]["model"] in [AzureModel]
        ]

    def test_azure_deployed_model_prompts(self):
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

                # self.assertIsInstance(prompt, dict)
                # self.assertIn("prompt", prompt)
                # self.assertIsInstance(prompt["prompt"], str)
                self.assertIsInstance(prompt, list)

                for message in prompt:
                    self.assertIsInstance(message, dict)
                    self.assertIn("role", message)
                    self.assertIsInstance(message["role"], str)
                    self.assertIn("content", message)
                    self.assertIsInstance(message["content"], (str, list))


class TestAzureDepModelConfig(unittest.TestCase):
    def test_azure_deployed_model_config(self):
        "Test if model config parameters passed as arguments are used"
        model = AzureModel(api_url="azure.llmebench.org", api_key="secret-key")

        self.assertEqual(model.api_url, "azure.llmebench.org")
        self.assertEqual(model.api_key, "secret-key")

    @patch.dict(
        "os.environ",
        {
            "AZURE_DEPLOYMENT_API_URL": "azure.llmebench.org",
            "AZURE_DEPLOYMENT_API_KEY": "secret-key",
        },
    )
    def test_azure_deployed_model_config_env_var(self):
        "Test if model config parameters passed as environment variables are used"
        model = AzureModel(api_url="azure.llmebench.org", api_key="secret-key")

        self.assertEqual(model.api_url, "azure.llmebench.org")
        self.assertEqual(model.api_key, "secret-key")

    @patch.dict(
        "os.environ",
        {
            "AZURE_DEPLOYMENT_API_URL": "petals.llmebench.org",
            "AZURE_DEPLOYMENT_API_KEY": "secret-key",
        },
    )
    def test_azure_deployed_model_config_priority(self):
        "Test if model config parameters passed as environment variables are used"
        model = AzureModel(api_url="azure.llmebench.org", api_key="secret-key")

        self.assertEqual(model.api_url, "azure.llmebench.org")
        self.assertEqual(model.api_key, "secret-key")
