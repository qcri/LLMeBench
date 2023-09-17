import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class SANADAlKhaleejDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(SANADAlKhaleejDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {"input": "some tweet", "label": "tech"}

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@article{einea2019sanad,
                title={Sanad: Single-label {A}rabic news articles dataset for automatic text categorization},
                author={Einea, Omar and Elnagar, Ashraf and Al Debsi, Ridhwan},
                journal={Data in brief},
                volume={25},
                pages={104076},
                year={2019},
                publisher={Elsevier}
            }""",
            "link": "https://data.mendeley.com/datasets/57zpx667y9/2",
            "license": "CC BY 4.0",
            "splits": {
                "test": "data/news_categorization/SANAD_alkhaleej_news_cat_test.tsv",
                "train": "data/news_categorization/SANAD_alkhaleej_news_cat_train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "politics",
                "religion",
                "medical",
                "sports",
                "tech",
                "finance",
                "culture",
            ],
        }

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        dir_path = "data/news_categorization/"
        for index, row in raw_data.iterrows():
            filename = row["file_path"].strip()
            file = open(dir_path + filename, "r")
            lines = file.readlines()
            lines = " ".join(lines).strip()
            label = row["class_label"]

            entry = {"input_id": filename, "input": lines, "label": label}
            data.append(entry)

        return data
