import csv

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class BanglaNewsCategorizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(BanglaNewsCategorizationDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "bn",
            "citation": """@article{alam2021review,
                              title={A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models},
                              author={Alam, Firoj and Hasan, Md Arid and Alam, Tanvir and Khan, Akib and Tajrin, Janntatul and Khan, Naira and Chowdhury, Shammur Absar},
                              journal={arXiv preprint arXiv:2107.03844},
                              year={2021}
                            }
                            @article{alam2020bangla,
                              title={Bangla Text Classification using Transformers},
                              author={Alam, Tanvirul and Khan, Akib and Alam, Firoj},
                              journal={arXiv preprint arXiv:2011.04446},
                              year={2020}
                            }
                            
                            @article{kunchukuttan2020ai4bharat,
                             author = {Anoop Kunchukuttan and Divyanshu Kakwani and Satish Golla and Gokul N.C. and Avik Bhattacharyya and Mitesh M. Khapra and Pratyush Kumar},
                             journal = {arXiv preprint arXiv:2005.00085},
                             title = {AI4Bharat-IndicNLP Corpus: Monolingual Corpora and Word Embeddings for Indic Languages},
                             year = {2020}
                            }""",
            "link": "https://github.com/banglanlp/bnlp-resources/tree/main/news_categorization",
            "license": "CC BY-NC-SA 2.0",
            "splits": {
                "test": "test.tsv",
                "train": "train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "entertainment",
                "state",
                "sports",
                "national",
                "kolkata",
                "international",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "News", "label": "entertainment", "id": 1}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            reader = csv.reader(fp, delimiter="\t")
            next(fp)
            for line_idx, line in enumerate(reader):
                content, label = line[0], line[1]  # line.strip().split("\t")
                label = label.capitalize()
                data.append({"input": content, "label": label, "id": line_idx + 1})

        return data
