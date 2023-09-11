import json
import logging
import random

from abc import ABC, abstractmethod

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS


class DatasetBase(ABC):
    """
    Base class for datasets

    Implementations of this class need to implement at least three mandatory methods;
    `metadata()`, `get_data_sample()` and `load_data()`. The purpose of objects of
    this class is to encapsulate all the subtleties and information for a specific
    dataset, and provide a consistent way for the framework to read the dataset.

    Attributes
    ----------
        None

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

    def __init__(self, **kwargs):
        pass

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
            "download_url" : str (optional)
                URL to data (for automatic downloads)
        """
        pass

    @abstractmethod
    def get_data_sample(self):
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

    def prepare_fewshots(self, target_data, train_data, n_shots, deduplicate=True):
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
        deduplicate : bool, defaults to True
            Whether the training samples should be de-duplicated (w.r.t test
            samples).

        Returns
        -------
        fewshot_data : generator
            A generator that returns `n_shots` train samples for every
            test sample
        """
        """"""

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
        # TODO: Need to handle not str inputs
        embedding_model = HuggingFaceEmbeddings(
            model_name="distiluse-base-multilingual-cased-v1"
        )
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
