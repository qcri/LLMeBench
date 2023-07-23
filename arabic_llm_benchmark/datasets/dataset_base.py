import json
import logging
import random

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

    @abstractmethod
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

    def stringify_sample(self, sample):
        new_sample = sample.copy()
        new_sample["input"] = json.dumps(new_sample["input"], ensure_ascii=False)
        return new_sample

    def destringify_sample(self, sample):
        new_sample = sample.copy()
        new_sample["input"] = json.loads(new_sample["input"])
        return new_sample

    def prepare_fewshots(self, target_data, train_data, n_shots, deduplicate=True):
        """Returns a generator for fewshot samples _per test instance_"""

        # Stringify inputs for few shot
        deserialization_required = False
        if not isinstance(train_data[0]["input"], str):
            logging.warning(
                "`input` is not a string, JSON stringifying for few shot preparation"
            )
            deserialization_required = True
            train_data = [self.stringify_sample(sample) for sample in train_data]

        # Remove empty inputs
        original_sample_count = len(train_data)
        train_data = [sample for sample in train_data if len(sample["input"]) > 0]
        filtered_sample_count = len(train_data)

        if filtered_sample_count < original_sample_count:
            logging.warning(
                f"Filtered out {original_sample_count - filtered_sample_count} due to empty input"
            )

        # Dedup train set against test set by doc ID before selecting examples
        # We discovered some datasets had overlap between train and test
        if deduplicate:
            original_sample_count = len(train_data)
            train_data = self.deduplicate_train_test(train_data, target_data)
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
                input_sample = self.stringify_sample(input_sample)
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
                examples = [self.destringify_sample(sample) for sample in examples]

            yield examples
