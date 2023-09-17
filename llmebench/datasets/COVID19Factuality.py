import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class COVID19FactualityDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(COVID19FactualityDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {"input": "some tweet", "label": "no"}

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{alam-etal-2021-fighting-covid,
                title = "Fighting the {COVID}-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society",
                author = "Alam, Firoj  and
                  Shaar, Shaden  and
                  Dalvi, Fahim  and
                  Sajjad, Hassan  and
                  Nikolov, Alex  and
                  Mubarak, Hamdy  and
                  Da San Martino, Giovanni  and
                  Abdelali, Ahmed  and
                  Durrani, Nadir  and
                  Darwish, Kareem  and
                  Al-Homaid, Abdulaziz  and
                  Zaghouani, Wajdi  and
                  Caselli, Tommaso  and
                  Danoe, Gijs  and
                  Stolk, Friso  and
                  Bruntink, Britt  and
                  Nakov, Preslav",
                booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
                month = nov,
                year = "2021",
                address = "Punta Cana, Dominican Republic",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2021.findings-emnlp.56",
                doi = "10.18653/v1/2021.findings-emnlp.56",
                pages = "611--649",
            }""",
            "license": "CC BY NC SA 4.0",
            "splits": {
                "test": "covid19_infodemic_arabic_data_factuality_binary_test.tsv",
                "train": "covid19_infodemic_arabic_data_factuality_binary_train.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["yes", "no"],
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["text"]
            tweet_id = row["tweet_id"]
            label = str(row["class_label"])
            data.append(
                {
                    "input": text,
                    "label": label,
                    "line_number": index,
                    "tweet_id": tweet_id,
                }
            )
        return data
