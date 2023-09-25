import os

import unittest

from llmebench import Benchmark


class TestAssetsTaskEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the benchmark assets
        benchmark = Benchmark(benchmark_dir="assets")
        cls.assets = benchmark.find_assets()

    def test_task_evaluation_failure(self):
        "Test if tasks used in assets handle failed runs"
        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]
                dataset_args = config.get("dataset_args", {})
                dataset_args["data_dir"] = ""
                dataset = config["dataset"](**dataset_args)
                data_sample = dataset.get_data_sample()

                task_args = config.get("task_args", {})
                task = config["task"](dataset=dataset, **task_args)
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

    def test_task_evaluation_format(self):
        "Test if evaluation function returns a dict"
        for asset in self.assets:
            with self.subTest(msg=asset["name"]):
                config = asset["config"]
                dataset_args = config.get("dataset_args", {})
                dataset_args["data_dir"] = ""
                dataset = config["dataset"](**dataset_args)
                data_sample = dataset.get_data_sample()

                task_args = config.get("task_args", {})
                task = config["task"](dataset=dataset, **task_args)
                evaluation_scores = task.evaluate(
                    [data_sample["label"]], [data_sample["label"]]
                )

                self.assertIsInstance(evaluation_scores, dict)
