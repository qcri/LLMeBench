from collections import defaultdict

from llmebench import Benchmark

import json

def main():
    benchmark = Benchmark(benchmark_dir="assets")

    assets = benchmark.find_assets()

    train_dataset_metadata = defaultdict(set)
    test_dataset_metadata = defaultdict(set)

    for asset in assets:
        configs = asset["module"].config()
        if isinstance(configs, dict):
            configs = [{"name": "dummy", "config": configs}]

        asset_name = asset["name"]
        asset_name = asset_name.replace("_BLOOMZ", "")
        asset_name = asset_name.replace("_GPT4", "")
        asset_name = asset_name.replace("_GPT35", "")
        asset_name = asset_name.replace("_ZeroShot", "")
        asset_name = asset_name.replace("_FewShot", "")
        
        for c in configs:
            config = c["config"]
            dataset_name = config["dataset"].__name__
            print(asset["name"])

            data_path = config["general_args"]["data_path"]
            if isinstance(data_path, dict):
                if "split" in data_path:
                    assert data_path["split"] == "test"
                    data_path = data_path["path"]
                else:
                    data_path = data_path["sentences_path"]

            train_data_path = None
            if "fewshot" in config["general_args"]:
                train_data_path = config["general_args"]["fewshot"]["train_data_path"]
            if isinstance(train_data_path, dict):
                if "split" in train_data_path:
                    assert train_data_path["split"] == "train" or train_data_path["split"] == "dev"
                    train_data_path = train_data_path["path"]
                else:
                    train_data_path = train_data_path["sentences_path"]

            test_dataset_metadata[dataset_name].add((data_path, asset_name))
            if train_data_path:
                train_dataset_metadata[dataset_name].add((train_data_path, asset_name))

    print("Test data paths")
    for dataset in test_dataset_metadata:
        print(dataset)
        if len(test_dataset_metadata[dataset]) == 1:
            continue
        for path, source in test_dataset_metadata[dataset]:
            print(f"\t{path} ({source})")

    print("\n\nTrain data paths")
    for dataset in train_dataset_metadata:
        print(dataset)
        if len(train_dataset_metadata[dataset]) == 1:
            continue
        for path, source in train_dataset_metadata[dataset]:
            print(f"\t{path} ({source})")



if __name__ == '__main__':
    main()