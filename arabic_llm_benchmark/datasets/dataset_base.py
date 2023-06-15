import logging
from abc import ABC, abstractmethod

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS


class DatasetBase(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def citation(self):
        pass

    @abstractmethod
    def get_data_sample(self):
        pass

    @abstractmethodgit
    def load_data(self, data_path, no_labels=False):
        """
        Returns a list of dictionaries,
        with at least the following keys:
                "input": <input-instance>
                "label": <label>
        The dictionaries can contain other keys as well
        which will be saved in the cache
        """
        pass

    def deduplicate_train_test(self, train_data, test_data):
        if len(test_data) > 0 and "input_id" not in test_data[0]:
            logging.warning(
                "`input_id` not found in data, no deduplication will be run"
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

    def prepare_fewshots(self, target_data, train_data, n_shots, deduplicate=True):
        """Returns a generator for fewshot samples _per test instance_"""

        # Dedup train set against test set by doc ID before selecting examples
        # We discovered some datasets had overlap between train and test
        if deduplicate:
            train_data = self.deduplicate_train_test(train_data, target_data)

        # TODO: MaxMarginalRelevanceExampleSelector should be generalized
        # TODO: Need to handle not str inputs
        embedding_model = HuggingFaceEmbeddings(
            model_name="distiluse-base-multilingual-cased-v1"
        )
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            train_data, embedding_model, FAISS, input_keys=["input"], k=n_shots
        )

        # TODO: Convert to an iterator
        # For each input sample, get few shot examples
        for idx, input_sample in enumerate(target_data):
            # Only need the input test content to select examples
            examples = example_selector.select_examples(input_sample)

            # Quick way to pre-compute examples for all test data at once.
            yield examples
