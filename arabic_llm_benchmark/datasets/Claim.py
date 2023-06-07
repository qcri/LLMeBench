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
                    author={Nakov, Preslav and Barr{\'o}n-Cede{\~n}o, Alberto and Da San Martino, Giovanni and Alam, Firoj and Kutlu, Mucahid and Zaghouani, Wajdi and Li, Chengkai and Shaar, Shaden and Mubarak, Hamdy and Nikolov, Alex},
                     year={2022},
                    booktitle={Proceedings of the Working Notes of CLEF 2022 - Conference and Labs of the Evaluation Forum}
                }"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": "1"}

    def load_data(self, data_path):
        formatted_data = []

        data = pd.read_csv(data_path, sep="\t")
        for index, tweet in data.iterrows():
            text = tweet["tweet_text"]
            label = str(tweet["class_label"])

            formatted_data.append({"input": text, "label": label, "line_number": index})

        return formatted_data

    """MARAM: Will all datasets have same format of train and test files? 
    maybe we need to implement a different class per split
    """
    def load_train_data(train_data_path):
        formatted_data = []

        data = pd.read_csv(train_data_path, sep="\t")
        for index, tweet in data.iterrows():
            text = tweet["tweet_text"]
            label = str(tweet["class_label"])

            formatted_data.append({"input": text, "label": label, "line_number": index})

        return formatted_data

    def get_fewshot_selector(train_data):
        # MARAM: langchain expects all values in input examples to be String, our line_index is integer
        train_data_formatted = [{key: str(val) for key, val in dict.items()} for dict in train_data]

        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            train_data_formatted,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1", cache_folder="assets/sentence-transformers_distiluse-base-multilingual-cased-v1/"),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=2  # MARAM: this should be a parameter to be passed as part of the benchmark config
        )

        return example_selector
