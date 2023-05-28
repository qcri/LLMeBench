import argparse

from glob import glob
from pathlib import Path

import importlib
import utils

class SingleTaskBenchmark(object):
	def __init__(self, config, prompt_fn, post_process_fn, cache_dir):
		self.dataset = config["dataset"](**config["dataset_args"])
		self.task = config["task"](self.dataset, **config["task_args"])
		self.model = config["model"](cache_dir=cache_dir, **config["model_args"])

		self.prompt_fn = prompt_fn
		self.post_process_fn = post_process_fn
		self.data_path = config["general_args"]["data_path"]

	def run_pipeline(self, sample_key, input_sample):
		prompt = self.prompt_fn(input_sample)
		model_output = self.model.run_model(sample_key, **prompt)
		filtered_output = self.post_process_fn(model_output)

		return filtered_output

	def run_benchmark(self):
		data = self.task.load_data(self.data_path)

		true_labels = []
		predictions = []

		for sample_idx, (input_sample, label) in enumerate(data):
			true_labels.append(label)
			predictions.append(self.run_pipeline(sample_idx, input_sample))

		evaluation_score = self.task.evaluate(true_labels, predictions)

		return evaluation_score

class Benchmark(object):
	def __init__(self, benchmark_dir):
		self.benchmark_dir = Path(benchmark_dir)

	def find_runs(self):
		runs = []
		match_str = str(self.benchmark_dir / "**" / "*.py")
		for run in glob(match_str, recursive=True):
			module_path = str(Path(run).resolve())
			module_name = Path(run).name
			runs.append(
				{
					"name": run[len(str(self.benchmark_dir))+1:run.rfind(".")],
					"path": run,
					"module": utils.import_source_file(Path(module_path), module_name)
				}
			)

		return runs


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("benchmark_dir", type=Path)
	parser.add_argument("results_dir", type=Path)
	args = parser.parse_args()

	benchmark = Benchmark(args.benchmark_dir)

	runs = benchmark.find_runs()

	for run in runs:
		name = run["name"]
		config = run["module"].config()
		prompt_fn = run["module"].prompt
		post_process_fn = run["module"].post_process

		task_benchmark = SingleTaskBenchmark(config, prompt_fn, post_process_fn, cache_dir=args.results_dir / name)

		print(task_benchmark.run_benchmark())

if __name__ == '__main__':
	main()