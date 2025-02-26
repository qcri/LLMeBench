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

from llmebench import asset_utils, utils


class SingleTaskBenchmark(object):
    def __init__(
        self,
        name,
        config,
        prompt_fn,
        post_process_fn,
        cache_dir,
        data_dir,
        ignore_cache=False,
        ignore_postprocessing=True,
        limit=-1,
        n_shots=0,
    ):
        self.name = name

        # Pipeline components
        self.dataset_args = config.get("dataset_args", {})
        self.data_dir = data_dir
        self.dataset_cls = config["dataset"]

        self.task_args = config.get("task_args", {})
        self.task_cls = config["task"]

        self.model_args = config.get("model_args", {})
        self.model_cls = config["model"]

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
        self.should_download = "custom_test_split" not in config

        self.zeroshot = True
        if utils.is_fewshot_asset(config, prompt_fn):
            self.zeroshot = False
            self.deduplicate = True
            self.fewshot_embedding_model_name = None
            self.train_data_paths = utils.get_data_paths(config, "train")

            assert len(self.data_paths) == len(
                self.train_data_paths
            ), "A train split must be provided for every test split being run"
            if "fewshot" in general_args:
                self.deduplicate = general_args["fewshot"].get("deduplicate", True)
                self.fewshot_embedding_model_name = general_args["fewshot"].get(
                    "embedding_model_name", None
                )

        self.limit = limit
        self.n_shots = n_shots

    def is_zeroshot(self):
        return self.zeroshot

    def initialize_pipeline(self):
        if "data_dir" not in self.dataset_args:
            self.dataset_args["data_dir"] = self.data_dir
        self.dataset = self.dataset_cls(**self.dataset_args)
        self.task = self.task_cls(dataset=self.dataset, **self.task_args)
        self.model = self.model_cls(**self.model_args)

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
        self.initialize_pipeline()

        base_name = self.name
        base_cache_dir = self.cache_dir

        # Download dataset if not already present on disk and custom splits are not specified
        if self.should_download:
            self.dataset.download_dataset(
                self.data_dir, default_url="https://llmebench.qcri.org/data/"
            )

        # Create sub-directory for few shot experiments
        if not self.is_zeroshot():
            base_name = f"{self.name}/{self.n_shots}_shot"
            base_cache_dir = self.cache_dir / f"{self.n_shots}_shot"

        all_task_results = {}
        for split_idx, (split_name, data_path) in enumerate(self.data_paths):
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
                train_split_name, train_data_path = self.train_data_paths[split_idx]
                train_data = self.dataset.load_data(train_data_path)

                few_shots_data = self.dataset.prepare_fewshots(
                    data,
                    train_data,
                    self.n_shots,
                    embedding_model_name=self.fewshot_embedding_model_name,
                    deduplicate=self.deduplicate,
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
                # logging.info(f"Running sample {sample_idx}: {input_sample['input']}")
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
    parser = utils.ArgumentParserWithDefaultSubcommand()
    parser.set_default_subparser("benchmark")
    subparsers = parser.add_subparsers(
        help="Defaults to 'benchmark'. Specify a command before the help flag to see detailed usage for each command.",
        dest="subparser_name",
    )

    parser_main = subparsers.add_parser("benchmark", help="Run the benchmark")

    parser_main.add_argument("benchmark_dir", type=Path)
    parser_main.add_argument("results_dir", type=Path)
    parser_main.add_argument(
        "-f",
        "--filter",
        default="*.py",
        help="Filter to match specific tasks in the benchmark."
        " Examples are '*ZeroShot*', 'Demography*', '*.py' (default)."
        " The .py extension is added automatically if missing.",
    )
    parser_main.add_argument("--ignore_cache", action="store_true")
    parser_main.add_argument(
        "-l",
        "--limit",
        default=-1,
        type=int,
        help="Limit the number of input instances that will be processed",
    )

    parser_main.add_argument(
        "-e", "--env", type=Path, help="Path to an .env file to load model parameters"
    )

    parser_main.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not run any actual models, but load all the data and process"
        " few shots. Existing cache will be ignored and overwritten.",
    )

    few_shot_args = parser_main.add_argument_group("Few Shot Experiments")
    few_shot_args.add_argument(
        "-n",
        "--n_shots",
        default=0,
        type=int,
        help="Number of samples to select for few shot learning."
        " Defaults to zero, i.e. Zero shot learning."
        " When this argument is 0, only zero shot assets will be run,"
        " and when it is non-zero, only few shot experiments will be run.",
    )

    parser_data = subparsers.add_parser("data", help="Dataset specific commands")

    data_subparsers = parser_data.add_subparsers(
        dest="data_subparser_name",
    )

    parser_data_download = data_subparsers.add_parser(
        "download", help="Download specific dataset"
    )

    parser_data_download.add_argument(
        "--download_server",
        type=str,
        default="https://llmebench.qcri.org/data/",
        help="URL to server containing dataset archives",
    )
    parser_data_download.add_argument(
        "dataset_name",
        type=str,
        help="Download the dataset with the given name (e.g Aqmar)",
    )

    parser_assets = subparsers.add_parser("assets", help="Assets specific commands")

    assets_subparsers = parser_assets.add_subparsers(
        dest="assets_subparser_name",
    )

    parser_assets_download = assets_subparsers.add_parser(
        "download",
        help="Download all assets. Will update if assets to latest version if they are already present.",
    )
    parser_assets_download.add_argument(
        "--work_dir",
        type=str,
        default="./",
        help="Default path for managing the benchmarking assets. Assets will be saved in `<work_dir>/assets`, and associated versioning files in `<work_dir>/.git`. Be default <work_dir> is set to the current working directory.",
    )

    # Common options
    for subparser in [parser_main, parser_data_download]:
        subparser.add_argument(
            "--data_dir",
            default="data/",
            type=Path,
            help="Default path for data. All relative paths will be resolved by"
            " using this as the base path",
        )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Handle downloading of datasets
    if args.subparser_name == "data":
        if args.data_subparser_name == "download":
            dataset_name = args.dataset_name
            if not dataset_name.endswith("Dataset"):
                dataset_name = f"{dataset_name}Dataset"
            try:
                mod = __import__("llmebench.datasets", fromlist=[dataset_name])
                dataset = getattr(mod, dataset_name)
            except AttributeError:
                logging.error(f"{dataset_name} not found in llmebench.datasets)")
                return
            dataset.download_dataset(args.data_dir, default_url=args.download_server)
        else:
            parser_data.print_help()
        return
    elif args.subparser_name == "assets":
        if args.assets_subparser_name == "download":
            asset_utils.download_all(args.work_dir)
        else:
            parser_assets.print_help()
        return

    # We must be performing benchmarking now and not any subcommands
    if args.env:
        load_dotenv(args.env, override=True)

    if args.benchmark_dir is None or args.results_dir is None:
        logging.error(parser.print_usage())
        logging.error(
            "The following arguments are required: benchmark_dir, results_dir"
        )
        return

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
                data_dir=args.data_dir,
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
