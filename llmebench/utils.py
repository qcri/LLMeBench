import argparse
import importlib.util
import sys

from inspect import signature

from pathlib import Path
from typing import TYPE_CHECKING


# https://stackoverflow.com/a/4575792
class ArgumentParserWithDefaultSubcommand(argparse.ArgumentParser):
    __default_subparser = None

    def set_default_subparser(self, name):
        self.__default_subparser = name

    def _parse_known_args(self, arg_strings, *args, **kwargs):
        in_args = set(arg_strings)
        d_sp = self.__default_subparser
        if d_sp is not None and not {"-h", "--help"}.intersection(in_args):
            for x in self._subparsers._actions:
                subparser_found = isinstance(
                    x, argparse._SubParsersAction
                ) and in_args.intersection(x._name_parser_map.keys())
                if subparser_found:
                    break
            else:
                # insert default in first position, this implies no
                # global options without a sub_parsers specified
                arg_strings = [d_sp] + arg_strings
        return super(ArgumentParserWithDefaultSubcommand, self)._parse_known_args(
            arg_strings, *args, **kwargs
        )


# https://stackoverflow.com/a/41595552
def import_source_file(fname: Path, modname: str) -> "types.ModuleType":
    """
    Import a Python source file and return the loaded module.

    Args:
        fname: The full path to the source file.  It may container characters like `.`
            or `-`.
        modname: The name for the loaded module.  It may contain `.` and even characters
            that would normally not be allowed (e.g., `-`).
    Return:
        The imported module

    Raises:
        ImportError: If the file cannot be imported (e.g, if it's not a `.py` file or if
            it does not exist).
        Exception: Any exception that is raised while executing the module (e.g.,
            :exc:`SyntaxError).  These are errors made by the author of the module!
    """
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(modname, fname)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{modname}' at: {fname}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {fname}") from e
    return module


def is_fewshot_asset(config, prompt_fn):
    """Detect if a given asset is zero shot or few show"""
    sig = signature(prompt_fn)
    general_args = config.get("general_args", {})
    return "fewshot" in general_args or len(sig.parameters) == 2


def get_data_paths(config, split):
    """Given a asset config, return the appropriate data paths"""
    assert split in ["train", "test"]

    dataset_args = config.get("dataset_args", {})
    dataset = config["dataset"](**dataset_args)

    if split == "test":
        data_args = config.get("general_args", {})
    elif split == "train":
        general_args = config.get("general_args", {})
        data_args = general_args.get("fewshot", {})

    is_generic_dataset = dataset.metadata().get("generic", False)

    data_paths = []
    if is_generic_dataset:
        # custom_test/train_split _must_ be present for generic datasets
        data_paths.append(("custom", data_args[f"custom_{split}_split"]))
    elif f"custom_{split}_split" in data_args:
        data_paths.append(("custom", data_args[f"custom_{split}_split"]))
    elif f"{split}_split" in data_args:
        requested_splits = data_args[f"{split}_split"]
        if not isinstance(requested_splits, list):
            requested_splits = [requested_splits]
        requested_splits = [rs.split("/") for rs in requested_splits]
        available_splits = dataset.metadata()["splits"]

        for requested_split in requested_splits:
            if len(requested_split) == 1:
                # Single level split like "test" or "ar"
                assert (
                    requested_split[0] in available_splits
                ), "Requested split not found in dataset"
                if split in available_splits[requested_split[0]]:
                    # Pick "test"/"train" automatically, if available
                    data_paths.append(
                        (
                            requested_split[0],
                            available_splits[requested_split[0]][split],
                        )
                    )
                else:
                    data_paths.append(
                        (requested_split[0], available_splits[requested_split[0]])
                    )
            else:
                # Multilevel split like "ar" -> "test"
                assert (
                    requested_split[0] in available_splits
                ), "Requested split not found in dataset"
                assert (
                    requested_split[1] in available_splits[requested_split[0]]
                ), "Requested split not found in dataset"
                data_paths.append(
                    (
                        f"{requested_split[0]}/{requested_split[1]}",
                        available_splits[requested_split[0]][requested_split[1]],
                    )
                )
    else:
        # Use default splits
        available_splits = dataset.metadata()["splits"]
        if "default" in available_splits:
            # Multilevel splits
            for av_split in available_splits["default"]:
                assert (
                    split in available_splits[av_split]
                ), f'No "{split}" split found in dataset, please specify split explicitly. Available splits are: {", ".join(available_splits[av_split])}'
                data_paths.append((av_split, available_splits[av_split][split]))
        else:
            # Single level splits
            assert (
                split in available_splits
            ), f'No "{split}" split found in dataset, please specify split explicitly. Available splits are: {", ".join(available_splits)}'
            data_paths.append((split, available_splits[split]))

    return data_paths


def resolve_path(path, dataset, data_dir):
    """Return absolute path"""
    if not isinstance(path, Path):
        path = Path(path)

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if not str(path).startswith(":data_dir:") and path.is_absolute():
        return path
    elif str(path).startswith(":data_dir:"):
        return data_dir / str(path)[len(":data_dir:") :]
    else:
        dataset_name = dataset.__class__.__name__
        if dataset_name.endswith("Dataset"):
            dataset_name = dataset_name[: -len("Dataset")]
        return data_dir / dataset_name / path
