import unittest
from unittest.mock import patch

from llmebench import Benchmark
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes

from llmebench.utils import is_fewshot_asset


class TestAssetsForHuggingFaceInferenceAPIPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the HuggingFaceInferenceAPI model
        cls.assets = [
            asset
            for asset in all_assets
            if asset["config"]["model"] in [HuggingFaceInferenceAPIModel]
        ]

    def test_huggingface_inference_api_prompts(self):
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

                self.assertIsInstance(prompt, dict)
                self.assertIn("inputs", prompt)

    def test_asset_config(self):
        "Test if all assets are providing the correct model_args"

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]
                model_args = config["model_args"]

                self.assertIsInstance(model_args, dict)
                self.assertIn("task_type", model_args)
                self.assertIsInstance(model_args["task_type"], HuggingFaceTaskTypes)
                self.assertIn("inference_api_url", model_args)


class TestHuggingFaceInferenceAPIConfig(unittest.TestCase):
    def test_huggingface_inference_api_config(self):
        "Test if model config parameters passed as arguments are used"
        model = HuggingFaceInferenceAPIModel("task", "url", api_token="secret-token")

        self.assertEqual(model.api_token, "secret-token")

    @patch.dict(
        "os.environ",
        {
            "HUGGINGFACE_API_TOKEN": "secret-token",
        },
    )
    def test_huggingface_inference_api_config_env_var(self):
        "Test if model config parameters passed as environment variables are used"
        model = HuggingFaceInferenceAPIModel("task", "url")

        self.assertEqual(model.api_token, "secret-token")

    @patch.dict(
        "os.environ",
        {
            "HUGGINGFACE_API_TOKEN": "secret-token",
        },
    )
    def test_huggingface_inference_api_config_priority(self):
        "Test if model config parameters passed as environment variables are used"
        model = HuggingFaceInferenceAPIModel("task", "url", api_token="secret-token-2")

        self.assertEqual(model.api_token, "secret-token-2")
