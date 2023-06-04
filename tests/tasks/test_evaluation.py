import os

import unittest

from unittest.mock import patch

from arabic_llm_benchmark import Benchmark


class TestAssetsTaskEvaluation(unittest.TestCase):
    @classmethod
    @patch("os.environ")
    def setUpClass(cls, os_env_mock):
        # Handle environment variables required at runtime
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"

        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        cls.assets = benchmark.find_runs()

    @patch("os.environ")
    def test_task_evaluation_failure(self, os_env_mock):
        "Test if tasks used in assets handle failed runs"
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"
        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["module"].config()
                dataset = config["dataset"](**config["dataset_args"])
                data_sample = dataset.get_data_sample()
                task = config["task"](dataset=None, **config["task_args"])
                try:
                    task.evaluate([data_sample["label"]], [None])
                except Exception as e:
                    self.fail(
                        f"{config['task']}'s evaluation does not handle"
                        + "`None` predictions, which may occur incase a "
                        + "component of the pipeline fails. The exact error"
                        + f" encountered was: \n {e}"
                        ""
                    )

    @patch("os.environ")
    def test_task_evaluation_format(self, os_env_mock):
        "Test if evaluation function returns a dict"
        os_env_mock.__getitem__.side_effect = lambda x: "test_str"
        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["module"].config()
                dataset = config["dataset"](**config["dataset_args"])
                data_sample = dataset.get_data_sample()
                task = config["task"](dataset=None, **config["task_args"])
                evaluation_scores = task.evaluate(
                    [data_sample["label"]], [data_sample["label"]]
                )

                self.assertIsInstance(evaluation_scores, dict)
