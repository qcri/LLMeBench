import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ASNDDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ASNDDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {"input": "some tweet", "label": "crime-war-conflict"}

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{chowdhury-etal-2020-improving-arabic,
                title = "Improving {A}rabic Text Categorization Using Transformer Training Diversification",
                author = "Chowdhury, Shammur Absar  and
                  Abdelali, Ahmed  and
                  Darwish, Kareem  and
                  Soon-Gyo, Jung  and
                  Salminen, Joni  and
                  Jansen, Bernard J.",
                booktitle = "Proceedings of the Fifth Arabic Natural Language Processing Workshop",
                month = dec,
                year = "2020",
                address = "Barcelona, Spain (Online)",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2020.wanlp-1.21",
                pages = "226--236",                
            }""",
            "link": "https://github.com/shammur/Arabic_news_text_classification_datasets/",
            "license": "CC BY 4.0",
            "splits": {
                "test": "data/news_categorization/Arabic_Social_Media_News_Dataset_ASND/sm_news_ar_tst.csv",
                "train": "data/news_categorization/Arabic_Social_Media_News_Dataset_ASND/sm_news_ar_trn.csv",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "crime-war-conflict",
                "spiritual",
                "health",
                "politics",
                "human-rights-press-freedom",
                "education",
                "business-and-economy",
                "art-and-entertainment",
                "others",
                "science-and-technology",
                "sports",
                "environment",
            ],
        }

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep=",", dtype={"ID": object})
        for index, row in raw_data.iterrows():
            id = row["ID"]
            text = row["Content"]
            label = row["Class"].lower()
            entry = {"input_id": id, "input": text, "label": label}
            data.append(entry)

        return data
