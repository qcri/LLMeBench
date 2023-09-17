import unittest
from unittest.mock import patch

from llmebench import Benchmark
from llmebench.models import PetalsModel

from llmebench.utils import is_fewshot_asset


class TestAssetsForPetalsPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        all_assets = benchmark.find_assets()

        # Filter out assets not using the Petals model
        cls.assets = [
            asset for asset in all_assets if asset["config"]["model"] in [PetalsModel]
        ]

    def test_petals_prompts(self):
        "Test if all assets using this model return data in an appropriate format for prompting"

        n_shots = 3  # Sample for few shot prompts

        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]
                dataset = config["dataset"](**config["dataset_args"])
                data_sample = dataset.get_data_sample()
                if is_fewshot_asset(config, asset["module"].prompt):
                    prompt = asset["module"].prompt(
                        data_sample["input"],
                        [data_sample for _ in range(n_shots)],
                    )
                else:
                    prompt = asset["module"].prompt(data_sample["input"])

                self.assertIsInstance(prompt, dict)
                self.assertIn("prompt", prompt)
                self.assertIsInstance(prompt["prompt"], str)


class TestPetalsConfig(unittest.TestCase):
    def test_petals_config(self):
        "Test if model config parameters passed as arguments are used"
        model = PetalsModel(api_url="petals.llmebench.org")

        self.assertEqual(model.api_url, "petals.llmebench.org")

    @patch.dict(
        "os.environ",
        {
            "PETALS_API_URL": "petals.llmebench.org",
        },
    )
    def test_petals_config_env_var(self):
        "Test if model config parameters passed as environment variables are used"
        model = PetalsModel()

        self.assertEqual(model.api_url, "petals.llmebench.org")

    @patch.dict(
        "os.environ",
        {
            "PETALS_API_URL": "petals.llmebench.org",
        },
    )
    def test_petals_config_priority(self):
        "Test if model config parameters passed as environment variables are used"
        model = PetalsModel(api_url="petals2.llmebench.org")

        self.assertEqual(model.api_url, "petals2.llmebench.org")
