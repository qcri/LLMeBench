import argparse

from glob import glob
from pathlib import Path

import importlib
import utils
import json

class SingleTaskBenchmark(object):
	def __init__(self, config, prompt_fn, post_process_fn, cache_dir, ignore_cache=False, ignore_postprocessing=True):
		# Pipeline components
		self.dataset = config["dataset"](**config["dataset_args"])
		self.task = config["task"](self.dataset, **config["task_args"])
		self.model = config["model"](**config["model_args"])

		# Caching parameters
		self.cache_dir = cache_dir
		if not self.cache_dir.exists():
			self.cache_dir.mkdir(parents=True)
		self.ignore_cache = ignore_cache
		self.ignore_post_processing = ignore_postprocessing

		# Model inference
		self.prompt_fn = prompt_fn
		self.post_process_fn = post_process_fn

		# Data parameters
		self.data_path = config["general_args"]["data_path"]

	def run_pipeline(self, sample_key, input_sample, cache_payload=None):
		# Prepare the prompt
		if "prompt" in cache_payload:
			prompt = cache_payload["prompt"]
		else:
			prompt = self.prompt_fn(input_sample)
			cache_payload["prompt"] = prompt

		# Run the model
		if "model_output" in cache_payload:
			model_output = cache_payload["model_output"]

		if "model_output" not in cache_payload or "response" not in model_output:
			model_output = self.model.run_model(**prompt)
			cache_payload["model_output"] = model_output

		if "response" not in model_output:
			return cache_payload

		if "filtered_output" in cache_payload and not self.ignore_post_processing:
			filtered_output = cache_payload["filtered_output"]
		else:
			filtered_output = self.post_process_fn(model_output["response"])
			cache_payload["filtered_output"] = filtered_output

		return cache_payload

	def run_benchmark(self):
		data = self.task.load_data(self.data_path)

		true_labels = []
		predictions = []

		for sample_idx, input_sample in enumerate(data):
			cache_path = self.cache_dir / f"{sample_idx}.json"
			true_labels.append(input_sample["label"])

			cache_payload = {"input": input_sample}
			if cache_path.exists() and not self.ignore_cache:
				with open(cache_path, "r") as fp:
					cache_payload = json.load(fp)

			cache_payload = self.run_pipeline(sample_idx, input_sample["input"], cache_payload)
			if "filtered_output" in cache_payload:
				predictions.append(cache_payload["filtered_output"])
			else:
				predictions.append(None)

			# Save the cache payload
			with open(cache_path, "w") as fp:
				json.dump(cache_payload, fp)

		evaluation_scores = self.task.evaluate(true_labels, predictions)

		return evaluation_scores

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