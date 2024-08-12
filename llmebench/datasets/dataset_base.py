import json
import logging
import os
import random

from abc import ABC, abstractmethod
from pathlib import Path

from pooch import Decompress, HTTPDownloader, Pooch, retrieve, Untar, Unzip

import llmebench.utils as utils


class DatasetBase(ABC):
    """
    Base class for datasets

    Implementations of this class need to implement at least three mandatory methods;
    `metadata()`, `get_data_sample()` and `load_data()`. The purpose of objects of
    this class is to encapsulate all the subtleties and information for a specific
    dataset, and provide a consistent way for the framework to read the dataset.

    Attributes
    ----------
    data_dir : str
        Base path of data containing all datasets. Defaults to "data" in the current
        working directory.

    Methods
    -------
    metadata():
        Returns metadata for the dataset

    get_data_sample():
        Returns one sample of data. Useful to see the structure of loaded data

    load_data(data_path="", no_labels=False):
        Loads data from the given path and returns a list of data samples

    prepare_fewshots(target_data=[], train_data=[], n_shots=1, deduplicate=True):
        Returns a generator that provides few shot samples for every test sample

    Notes
    -----
    - Consider overriding `_deduplicate_train_test` to replace the default "input_id"
    based de-duplication between train/test
    - If the data is not JSON serializable, `_stringify_sample`/`_destringify_sample`
    must be re-implemented to provide serialization/deserialization of samples. This is
    primarily used for some fewshot sampling methods.

    """

    def __init__(self, data_dir="data/", **kwargs):
        self.data_dir = data_dir

    @staticmethod
    @abstractmethod
    def metadata():
        """
        Returns the dataset's metadata

        Arguments
        ---------
            None

        Returns
        -------
        metadata : dict
            The returned dictionary _must_ have the following keys:
            "citation" : str
                bib-formatted citation for the dataset
            "language" : str|list
                Can be one of:
                    "multilingual"
                    ["ar", "fr", "en"] # List of supported languages
                    "ar" # Single supported language
                Languages should be identified by their IETF language tags
            The returned dictionary _can_ have the following additional keys:
            "link" : str
                Link to the representative page for the dataset
            "license" : str
                Original license under which the dataset was released
            "splits" : dict
                A dictionary containing the keys "test", "dev" and "train"
                (at least one). "test" will be used automatically for
                evaluation, if present. Asset can specify a different split
                if necessary. Multiple splits are also supported, by having
                a nested dictionary structure, where the first level should
                be the split name, and the second level should include the
                actual "test"/"dev"/"train" splits. A special "default" split
                can also be included, whose value must be a list of split
                names that will be run by default.
            "task_type" : llmebench.tasks.TaskType
                The type of task this dataset targets. Used by the Random
                Model.
            "class_labels" : list (optional)
                List of class labels, must be provided when `task_type` is
                `Classification`, `MultiLabelClassification` or
                `SequenceLabeling`.
            "score_range" : tuple (optional)
                Score range defining (min_val, max_val). Must be defined
                when `task_type` is `Regression`
            "download_url" : str (optional)
                URL to data (for automatic downloads)
        """
        pass

    @staticmethod
    @abstractmethod
    def get_data_sample():
        """
        Returns a single data sample.

        This function is useful to understand the structure of the underlying
        data. All loaded samples _must_ match this sample.

        Arguments
        ---------
            None

        Returns
        -------
        sample : dict
            _Must_ contain at least two keys "input" and "label".
            "input_id" can be specified to help with de-duplication
            between train/dev/test data. Can include additional keys.
        """
        pass

    @abstractmethod
    def load_data(self, data_path, no_labels=False):
        """
        Load data from data_path.

        Arguments
        ---------
        data_path : str|list|dict
            Path to dataset. Can be a list or dict as well.
        no_labels : bool
            Specifies if the data_path has a split with no labels

        Returns
        -------
        data : list
            List of dictionaries, where each dictionary is structured like
            `get_data_sample()`'s output
        """
        pass

    def _deduplicate_train_test(self, train_data, test_data):
        """
        Filter train data to avoid overlap with test data

        The default implementation de-duplicates based on an "input_id"
        element in the sample dictionary.

        Arguments
        ---------
        train_data : list
            Loaded train data
        test_data : list
            Loaded test data

        Returns
        -------
        filtered_train_data : list
            Train data with overlapping test samples removed
        """
        if len(test_data) > 0 and "input_id" not in test_data[0]:
            logging.warning(
                "`input_id` not found in data, no de-duplication will be run"
            )
            # TODO: Add fallback to input, label deep comparison
            return train_data

        final_train_data = []
        test_ids = [tid["input_id"] for tid in test_data]

        for train_sample in train_data:
            if train_sample["input_id"] in test_ids:
                continue
            final_train_data.append(train_sample)

        return final_train_data

    def _stringify_sample(self, sample):
        """
        Serialize data sample into a string.

        Primarily used for some fewshot samplers that work only on strings.
        By default uses JSON serialization; If the data is not JSON serializable,
        this function must be re-implemented in the implementing class.

        Arguments
        ---------
        sample : dict
            Input sample, with the same structure as that returned by
            `get_data_sample()`

        Returns
        -------
        new_sample : dict
            Same as the input sample, except the value associated with the key
            "input" must be a string
        """
        new_sample = sample.copy()
        new_sample["input"] = json.dumps(new_sample["input"], ensure_ascii=False)
        return new_sample

    def _destringify_sample(self, sample):
        """
        Deserialize data sample from a string.

        Primarily used for some fewshot samplers that work only on strings.
        By default uses JSON deserialization; If the data is not JSON deserializable,
        this function must be re-implemented in the implementing class.

        Arguments
        ---------
        sample : dict
            Output of `_stringify_sample()`

        Returns
        -------
        new_sample : dict
            Sample with the same structure as that returned by
            `get_data_sample()`
        """
        new_sample = sample.copy()
        new_sample["input"] = json.loads(new_sample["input"])
        return new_sample

    def prepare_fewshots(
        self,
        target_data,
        train_data,
        n_shots,
        embedding_model_name=None,
        deduplicate=True,
    ):
        """
        Returns a generator for fewshot samples _per test instance_

        Arguments
        ---------
        target_data : list
            Test samples
        train_data : list
            Train/Dev samples to pick few shot samples from
        n_shots : int
            Number of samples to pick for each test sample
        embedding_model_name : str
            The model to use for extracting embeddings to use for similarity computation.
            Defaults to 'distiluse-base-multilingual-cased-v1'
        deduplicate : bool, defaults to True
            Whether the training samples should be de-duplicated (w.r.t test
            samples).

        Returns
        -------
        fewshot_data : generator
            A generator that returns `n_shots` train samples for every
            test sample
        """

        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.prompts.example_selector import (
            MaxMarginalRelevanceExampleSelector,
        )
        from langchain.vectorstores import FAISS

        if embedding_model_name is None:
            embedding_model_name = "distiluse-base-multilingual-cased-v1"

        # Stringify inputs for few shot
        deserialization_required = False
        if not isinstance(train_data[0]["input"], str):
            logging.warning(
                "`input` is not a string, JSON stringifying for few shot preparation"
            )
            deserialization_required = True
            train_data = [self._stringify_sample(sample) for sample in train_data]

        # Remove empty inputs
        original_sample_count = len(train_data)
        train_data = [
            sample for sample in train_data if len(sample["input"].strip()) > 0
        ]
        filtered_sample_count = len(train_data)

        if filtered_sample_count < original_sample_count:
            logging.warning(
                f"Filtered out {original_sample_count - filtered_sample_count} due to empty input"
            )

        # Dedup train set against test set by doc ID before selecting examples
        # We discovered some datasets had overlap between train and test
        if deduplicate:
            original_sample_count = len(train_data)
            train_data = self._deduplicate_train_test(train_data, target_data)
            filtered_sample_count = len(train_data)
            if filtered_sample_count < original_sample_count:
                logging.warning(
                    f"Filtered out {original_sample_count - filtered_sample_count} due to duplication with test set"
                )

        # TODO: MaxMarginalRelevanceExampleSelector should be generalized
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            train_data, embedding_model, FAISS, input_keys=["input"], k=n_shots
        )

        # For each input sample, get few shot examples
        for idx, input_sample in enumerate(target_data):
            if deserialization_required:
                input_sample = self._stringify_sample(input_sample)
            if len(input_sample["input"].strip()) > 0:
                examples = example_selector.select_examples(input_sample)
            else:
                # Randomly select some train samples
                logging.warning(
                    f"Sample with empty input encountered, will pick few shot samples randomly from train"
                )
                examples = random.sample(train_data, k=n_shots)

            if deserialization_required:
                # Deserialize example
                examples = [self._destringify_sample(sample) for sample in examples]

            yield examples

    @classmethod
    def download_dataset(cls, data_dir, download_url=None, default_url=None):
        """
        Utility method to download a dataset if not present locally on disk.
        Can handle datasets of types *.zip, *.tar, *.tar.gz, *.tar.bz2, *.tar.xz.

        Arguments
        ---------
        download_url : str
            The url to the dataset. If not provided, falls back to the `download_url`
            provided by the Dataset's metadata. If missing, falls back to a default
            server specified by the `default_url` argument
        default_url : str
            Default server url to fall back to incase of missing download_urls

        Returns
        -------
        download_succeeded : bool
            Returns True if the dataset is already present on disk, or if download +
            extraction was successful.
        """

        dataset_name = cls.__name__
        if dataset_name.endswith("Dataset"):
            dataset_name = dataset_name[: -len("Dataset")]

        def decompress(fname, action, pup):
            """
            Post-processing hook to automatically detect the type of archive and
            call the correct processor (UnZip, Untar, Decompress)

            Arguments
            ---------
            fname : str
               Full path of the zipped file in local storage
            action : str
               One of "download" (file doesn't exist and will download),
               "update" (file is outdated and will download), and
               "fetch" (file exists and is updated so no download).
            pup : Pooch
               The instance of Pooch that called the processor function.

            Returns
            -------
            fnames : list
               List of all extracted files

            """
            # Default where the downloaded file is not a container/archive
            fnames = [fname]

            extract_dir = dataset_name

            if fname.endswith(".tar.xz"):
                extractor = Decompress(name=fname[:-3])
                fname = extractor(fname, action, pup)

                extractor = Untar(extract_dir=extract_dir)
                fnames = extractor(fname, action, pup)
            elif fname.endswith(".tar.bz2"):
                extractor = Decompress(name=fname[:-4])
                fname = extractor(fname, action, pup)

                extractor = Untar(extract_dir=extract_dir)
                fnames = extractor(fname, action, pup)
            elif fname.endswith(".tar.gz"):
                extractor = Decompress(name=fname[:-3])
                fname = extractor(fname, action, pup)

                extractor = Untar(extract_dir=extract_dir)
                fnames = extractor(fname, action, pup)
            elif fname.endswith(".xz"):
                extractor = Decompress(name=fname[:-3])
                fname = extractor(fname, action, pup)
                fnames = [fname]
            elif fname.endswith(".bz2"):
                extractor = Decompress(name=fname[:-4])
                fname = extractor(fname, action, pup)
                fnames = [fname]
            elif fname.endswith(".gz"):
                extractor = Decompress(name=fname[:-3])
                fname = extractor(fname, action, pup)
                fnames = [fname]
            elif fname.endswith(".tar"):
                extractor = Untar(extract_dir=extract_dir)
                fnames = extractor(fname, action, pup)
            elif fname.endswith(".zip"):
                extractor = Unzip(extract_dir=extract_dir)
                fnames = extractor(fname, action, pup)

            return fnames

        # Priority:
        #   Fn Argument
        #   Dataset metadata["download_url"]
        #   default_url/Dataset_name.zip
        download_urls = []
        if download_url is not None:
            download_urls.append(download_url)

        metadata_url = cls.metadata().get("download_url", None)
        if metadata_url is not None:
            download_urls.append(metadata_url)
        else:
            logging.warning(
                f"No default download url specified for {dataset_name}, will try to download from LLMeBench servers."
            )

        if default_url is not None:
            if default_url.endswith("/"):
                default_url = default_url[:-1]
            default_url = f"{default_url}/{dataset_name}.zip"
            download_urls.append(default_url)

        # Try downloading from available links in order of priority
        for download_url in download_urls:
            extension = ".zip"
            supported_extensions = [
                ".tar.xz",
                ".tar.bz2",
                ".tar.gz",
                ".xz",
                ".bz2",
                ".gz",
                ".tar",
                ".zip",
            ]

            for ext in supported_extensions:
                if download_url.endswith(ext):
                    extension = ext
                    break
            try:
                logging.info(f"Trying to fetch from {download_url}")
                if (Path(data_dir) / f"{dataset_name}{extension}").exists():
                    logging.info(f"Cached dataset found")
                    return True
                retrieve(
                    download_url,
                    known_hash=None,
                    fname=f"{dataset_name}{extension}",
                    path=data_dir,
                    progressbar=True,
                    processor=decompress,
                    downloader=HTTPDownloader(
                        headers={"User-Agent": "curl/8.1.2", "Accept": "*/*"}
                    ),
                )
                # If it was a *.tar.* file, we can safely delete the
                # intermediate *.tar file
                if extension in supported_extensions[:3]:
                    tar_file_path = Path(data_dir) / f"{dataset_name}.tar"
                    tar_file_path.unlink()
                logging.info(f"Fetch successful")
                return True
            except Exception as e:
                logging.warning(f"Failed to download: {e}")
                continue

        logging.warning(
            f"Failed to download dataset, tried the following urls: {', '.join(download_urls)}"
        )

        return False

    def resolve_path(self, path):
        return utils.resolve_path(path, self, self.data_dir)
