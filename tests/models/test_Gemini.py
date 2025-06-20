import unittest
from unittest.mock import patch

from llmebench import Benchmark
from llmebench.models import GeminiModel

from llmebench.utils import is_fewshot_asset


class TestAssetsForGeminiDepModelPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the Petals model
        cls.assets = [
            asset for asset in all_assets if asset["config"]["model"] in [GeminiModel]
        ]

    def test_gemini_deployed_model_prompts(self):
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


class TestGeminiDepModelConfig(unittest.TestCase):
    def test_gemini_deployed_model_config(self):
        "Test if model config parameters passed as arguments are used"
        model = GeminiModel(
            project_id="test_project_id",
            model_name="gemini-test",
            location="us-central1",
        )

        self.assertEqual(model.project_id, "test_project_id")
        self.assertEqual(model.location, "us-central1")
        self.assertEqual(model.model_name, "gemini-test")

    @patch.dict(
        "os.environ",
        {
            "GOOGLE_PROJECT_ID": "test_project_id",
            "LOCATION": "us-central1",
            "MODEL": "gemini-test",
        },
    )
    def test_gemini_deployed_model_config_env_var(self):
        "Test if model config parameters passed as environment variables are used"
        model = GeminiModel()

        self.assertEqual(model.project_id, "test_project_id")
        self.assertEqual(model.location, "us-central1")
        self.assertEqual(model.model_name, "gemini-test")

    @patch.dict(
        "os.environ",
        {
            "GOOGLE_PROJECT_ID": "test_project_id",
            "LOCATION": "us-central1",
            "MODEL": "gemini-test",
        },
    )
    def test_gemini_deployed_model_config_priority(self):
        "Test if model config parameters passed directly get priority"
        model = GeminiModel(
            project_id="test_project_id",
            model_name="gemini_test",
            location="us-central1",
        )

        self.assertEqual(model.project_id, "test_project_id")
        self.assertEqual(model.location, "us-central1")
        self.assertEqual(model.model_name, "gemini_test")
