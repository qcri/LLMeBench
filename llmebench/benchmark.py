import argparse

import importlib
import json
import logging
import sys
import traceback

from fnmatch import fnmatch
from glob import glob
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv

from llmebench import utils


class SingleTaskBenchmark(object):
    def __init__(
        self,
        name,
        config,
        prompt_fn,
        post_process_fn,
        cache_dir,
        ignore_cache=False,
        ignore_postprocessing=True,
        limit=-1,
        n_shots=0,
    ):
        self.name = name

        # Pipeline components
        dataset_args = config.get("dataset_args", {})
        self.dataset = config["dataset"](**dataset_args)

        task_args = config.get("task_args", {})
        self.task = config["task"](dataset=self.dataset, **task_args)

        model_args = config.get("model_args", {})
        self.model = config["model"](**model_args)

        general_args = config.get("general_args", {})

        # Caching parameters
        self.cache_dir = cache_dir
        self.ignore_cache = ignore_cache
        self.ignore_post_processing = ignore_postprocessing

        # Model inference
        self.prompt_fn = prompt_fn
        self.post_process_fn = post_process_fn

        # Data parameters
        self.data_paths = utils.get_data_paths(config, "test")

        self.zeroshot = True
        if utils.is_fewshot_asset(config, prompt_fn):
            self.zeroshot = False
            self.deduplicate = True
            self.train_data_paths = utils.get_data_paths(config, "train")
            if "fewshot" in general_args:
                self.deduplicate = general_args["fewshot"].get("deduplicate", True)

        self.limit = limit
        self.n_shots = n_shots

    def is_zeroshot(self):
        return self.zeroshot

    def run_pipeline(
        self,
        sample_key,
        input_sample,
        few_shot_examples,
        cache_payload=None,
        dry_run=False,
    ):
        summarized_payload = {}

        # Prepare the prompt
        if "prompt" in cache_payload:
            logging.info(f"\tLoading prompt from cache")
            prompt = cache_payload["prompt"]
        else:
            logging.info(f"\tGenerating prompt")
            if few_shot_examples:
                prompt = self.prompt_fn(input_sample, few_shot_examples)
            else:
                prompt = self.prompt_fn(input_sample)
            cache_payload["prompt"] = prompt

        if dry_run:
            return cache_payload, summarized_payload

        # Run the model
        if "model_output" in cache_payload:
            logging.info(f"\tLoading model output from cache")
            model_output = cache_payload["model_output"]

        if "model_output" not in cache_payload or "response" not in model_output:
            logging.info(f"\tRunning model")
            model_output = self.model.run_model(prompt)
            cache_payload["model_output"] = model_output

        if "response" not in model_output:
            summarized_payload["model_output"] = model_output["failure_exception"]
            return cache_payload, summarized_payload
        else:
            summarized_payload["model_output"] = self.model.summarize_response(
                model_output["response"]
            )

        if "filtered_output" in cache_payload and not self.ignore_post_processing:
            logging.info(f"\tLoading post processed output from cache")
            filtered_output = cache_payload["filtered_output"]
            summarized_payload["filtered_output"] = filtered_output
        else:
            logging.info(f"\tPost processing model outputs")
            try:
                filtered_output = self.post_process_fn(model_output["response"])
                if "filtered_output_failure_message" in cache_payload:
                    del cache_payload["filtered_output_failure_message"]
                cache_payload["filtered_output"] = filtered_output
                summarized_payload["filtered_output"] = filtered_output
            except Exception as e:
                exc_info = sys.exc_info()
                exception_str = "".join(traceback.format_exception(*exc_info))
                cache_payload["filtered_output_failure_message"] = exception_str
                summarized_payload["filtered_output"] = cache_payload[
                    "filtered_output_failure_message"
                ]

        return cache_payload, summarized_payload

    def run_benchmark(self, dry_run=False):
        base_name = self.name
        base_cache_dir = self.cache_dir

        # Create sub-directory for few shot experiments
        if not self.is_zeroshot():
            base_name = f"{self.name}/{self.n_shots}_shot"
            base_cache_dir = self.cache_dir / f"{self.n_shots}_shot"

        all_task_results = {}
        for split_name, data_path in self.data_paths:
            name = base_name
            cache_dir = base_cache_dir
            if len(self.data_paths) > 1:
                name = f"{self.name}/{split_name}"
                cache_dir = cache_dir / split_name

            # Create parent directory
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True)

            # Local cache
            full_summary_path = cache_dir / "summary.jsonl"
            failed_summary_path = cache_dir / "summary_failed.jsonl"

            data = self.dataset.load_data(data_path)
            few_shots_data = []
            if not self.zeroshot:
                train_data = []
                for split_name, train_data_path in self.train_data_paths:
                    train_data += self.dataset.load_data(train_data_path)

                few_shots_data = self.dataset.prepare_fewshots(
                    data, train_data, self.n_shots, deduplicate=self.deduplicate
                )

            true_labels = []
            predictions = []

            num_processed = 0
            full_summary_fp = open(full_summary_path, "w")

            num_failed = 0
            failed_summary_fp = open(failed_summary_path, "w")

            for sample_idx, (input_sample, few_shot_examples) in enumerate(
                zip_longest(data, few_shots_data, fillvalue=None)
            ):
                if self.limit > 0 and sample_idx >= self.limit:
                    break
                logging.info(f"Running sample {sample_idx}: {input_sample['input']}")
                num_processed += 1
                cache_path = cache_dir / f"{sample_idx}.json"
                true_labels.append(input_sample["label"])

                cache_payload = {"input": input_sample}

                if few_shot_examples is not None:
                    cache_payload["few_shot_examples"] = few_shot_examples

                if cache_path.exists() and not self.ignore_cache and not dry_run:
                    with open(cache_path, "r") as fp:
                        cache_payload = json.load(fp)

                summarized_payload = {
                    "input": input_sample["input"],
                    "label": input_sample["label"],
                }

                cache_payload, partial_summarized_payload = self.run_pipeline(
                    sample_idx,
                    input_sample["input"],
                    few_shot_examples,
                    cache_payload,
                    dry_run,
                )

                summarized_payload.update(partial_summarized_payload)

                if "filtered_output" in cache_payload:
                    predictions.append(cache_payload["filtered_output"])
                    full_summary_fp.write(
                        json.dumps(summarized_payload, ensure_ascii=False) + "\n"
                    )
                else:
                    if not dry_run:
                        logging.error(f"\tNo prediction for sample")
                        num_failed += 1
                    predictions.append(None)
                    full_summary_fp.write(
                        json.dumps(summarized_payload, ensure_ascii=False) + "\n"
                    )
                    failed_summary_fp.write(
                        json.dumps(summarized_payload, ensure_ascii=False) + "\n"
                    )

                # Save the cache payload
                with open(cache_path, "w") as fp:
                    json.dump(cache_payload, fp, ensure_ascii=False)

            full_summary_fp.close()
            failed_summary_fp.close()

            if num_failed > 0:
                logging.error(
                    f"{num_failed}/{len(data)} samples do not have any predictions"
                )
            evaluation_scores = self.task.evaluate(true_labels, predictions)

            # Prepare results
            task_results = {
                "num_processed": num_processed,
                "num_failed": num_failed,
                "evaluation_scores": evaluation_scores,
            }
            logging.info(f"{name}: {task_results['evaluation_scores']}")

            task_result_path = cache_dir / "results.json"

            with open(task_result_path, "w") as fp:
                json.dump(task_results, fp, ensure_ascii=False)

            all_task_results[name] = task_results

        return all_task_results


class Benchmark(object):
    def __init__(self, benchmark_dir):
        self.benchmark_dir = Path(benchmark_dir)

    def find_assets(self, filter_str="*.py"):
        if not filter_str.startswith("*"):
            filter_str = f"*{filter_str}"

        if not filter_str.endswith(".py") and not filter_str.endswith("*"):
            filter_str = f"{filter_str}*"

        assets = []
        match_str = str(self.benchmark_dir / "**" / "*.py")
        for asset in glob(match_str, recursive=True):
            module_path = str(Path(asset).resolve())
            module_name = Path(asset).name
            asset_name = asset[len(str(self.benchmark_dir)) + 1 : asset.rfind(".")]

            if not fnmatch(module_path.lower(), filter_str.lower()):
                logging.info(
                    f"Skipping {asset[len(str(self.benchmark_dir)) + 1 :]} because of --filter"
                )
                continue

            # Search for sub-assets
            asset_module = utils.import_source_file(Path(module_path), module_name)

            config = asset_module.config()

            if isinstance(config, dict):
                # Single config
                assets.append(
                    {
                        "name": asset_name,
                        "path": asset,
                        "module": asset_module,
                        "config": config,
                    }
                )
            elif isinstance(config, list):
                # Multi config
                for subconfig in config:
                    assets.append(
                        {
                            "name": f"{asset_name}/{subconfig['name']}",
                            "path": asset,
                            "module": asset_module,
                            "config": subconfig["config"],
                        }
                    )
            else:
                raise ValueError("Invalid configuration")

        return assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "-f",
        "--filter",
        default="*.py",
        help="Filter to match specific tasks in the benchmark."
        " Examples are '*ZeroShot*', 'Demography*', '*.py' (default)."
        " The .py extension is added automatically if missing.",
    )
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument(
        "-l",
        "--limit",
        default=-1,
        type=int,
        help="Limit the number of input instances that will be processed",
    )

    parser.add_argument(
        "-e", "--env", type=Path, help="Path to an .env file to load model parameters"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run any actual models, but load all the data and process few shots. Existing cache will be ignored and overwritten.",
    )

    group = parser.add_argument_group("Few Shot Experiments")
    group.add_argument(
        "-n",
        "--n_shots",
        default=0,
        type=int,
        help="Number of samples to select for few shot learning."
        " Defaults to zero, i.e. Zero shot learning."
        " When this argument is 0, only zero shot assets will be run,"
        " and when it is non-zero, only few shot experiments will be run.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.env:
        load_dotenv(args.env)

    benchmark = Benchmark(args.benchmark_dir)

    assets = benchmark.find_assets(filter_str=args.filter)

    if not args.results_dir.exists():
        args.results_dir.mkdir(parents=True)

    all_results_path = args.results_dir / "all_results.json"

    if not all_results_path.exists():
        with open(all_results_path, "w") as fp:
            json.dump({}, fp)

    with open(all_results_path, "r") as fp:
        all_results = json.load(fp)

    for asset in assets:
        name = asset["name"]
        config = asset["config"]
        prompt_fn = asset["module"].prompt
        post_process_fn = asset["module"].post_process

        try:
            logging.info(f"Running benchmark: {name}")
            task_benchmark = SingleTaskBenchmark(
                name,
                config,
                prompt_fn,
                post_process_fn,
                cache_dir=args.results_dir / name,
                ignore_cache=args.ignore_cache,
                limit=args.limit,
                n_shots=args.n_shots,
            )

            if task_benchmark.is_zeroshot() and args.n_shots > 0:
                logging.warning(
                    f"{name}: Skipping because asset is zero shot and --n_shots is non zero"
                )
                continue

            if not task_benchmark.is_zeroshot() and args.n_shots == 0:
                logging.warning(
                    f"{name}: Skipping because asset is few shot and --n_shots is zero"
                )
                continue

            all_task_results = task_benchmark.run_benchmark(dry_run=args.dry_run)
            for task_name in all_task_results:
                task_results = all_task_results[task_name]

                all_results[task_name] = task_results
        except Exception as e:
            logging.error(f"{name} failed to run")
            traceback.print_exc()

    with open(all_results_path, "w") as fp:
        json.dump(all_results, fp, ensure_ascii=False)
