import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase
from langchain.prompts.example_selector import MaxMarginalRelevanceExampleSelector
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class CovidClaimDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CovidClaimDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{nakov2022overview,
                    title={Overview of the CLEF-2022 CheckThat! lab task 1 on identifying relevant claims in tweets},
                    author={Nakov, Preslav and Barr{\\o}n-Cede{\\~n}o, Alberto and Da San Martino, Giovanni and Alam, Firoj and Kutlu, Mucahid and Zaghouani, Wajdi and Li, Chengkai and Shaar, Shaden and Mubarak, Hamdy and Nikolov, Alex},
                     year={2022},
                    booktitle={Proceedings of the Working Notes of CLEF 2022 - Conference and Labs of the Evaluation Forum}
                }"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": "1"}

    def load_data(self, data_path):
        formatted_data = []

        with open(data_path,"r",encoding='utf-8') as in_file:
            next(in_file)
            for index,line in enumerate(in_file):
                tweet = [str(s.strip()) for s in line.split("\t")]

                text = tweet[3]
                label = tweet[4]
                twt_id = tweet[1]

                formatted_data.append({"input": text, "label": label, "line_number": index, "input_id": twt_id})

        return formatted_data

    def load_train_data(self, train_data_path):
        formatted_data = []

        with open(train_data_path, "r", encoding='utf-8') as in_file:
            next(in_file)
            for index, line in enumerate(in_file):
                tweet = [str(s.strip()) for s in line.split("\t")]
                text = tweet[3]
                label = tweet[4]
                twt_id = tweet[1]

                # For later on in FS, langchain expect all values to be strings
                formatted_data.append({"input": text, "label": label, "line_number": str(index), "input_id": twt_id})

        return formatted_data


    def dedup_train_test(self, target_data, train_data):
        final_train_data = []
        test_ids = [tid['input_id'] for tid in target_data]

        for train_sample in train_data:
            if train_sample['input_id'] in test_ids: continue
            final_train_data.append(train_sample)

        return final_train_data

    # MARAM: this function should be called once as we do when we load train data
    # MARAM: n_shots should be a parameter passed form the asset
    def prepare_fewshots(self, target_data, train_data, n_shots):
        few_shots_per_input = {}

        print(len(train_data))
        # Dedup train set against test set by doc ID before selecting examples
        # We discovered some datasets had overlap between train and test
        train_data = self.dedup_train_test(target_data, train_data)
        print(len(train_data))

        # MARAM: MaxMarginalRelevanceExampleSelector should be a variable
        # Cache is used to store the bert model we are using
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(train_data, HuggingFaceEmbeddings(
            model_name="distiluse-base-multilingual-cased-v1",
            cache_folder="assets/sentence-transformers_distiluse-base-multilingual-cased-v1/"),
                                                                             FAISS, k=n_shots)

        # TODO: Should be an iterator
        # For each input sample, get few shot examples
        for input_sample in target_data:
            input_sample_content = input_sample['input']
            # Only need the input test content to select examples
            examples = example_selector.select_examples({"input": input_sample_content, "label": ""})

            # Quick way to pre-compute examples for all test data at once.
            few_shots_per_input[input_sample_content] = examples

        return few_shots_per_input