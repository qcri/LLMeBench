import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class CT22AttentionworthyDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CT22AttentionworthyDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {"input": "some tweet", "label": "no_not_interesting"}

    @staticmethod
    def metadata():
        return {
            "language": ["ar", "bg", "nl", "en", "tr"],
            "citation": """@InProceedings{clef-checkthat:2022:task1,
                author = {Nakov, Preslav and Barr\\'{o}n-Cede\\~{n}o, Alberto and Da San Martino, Giovanni and Alam, Firoj and M\\'{\\i}guez, Rub\'{e}n and Caselli, Tommaso and Kutlu, Mucahid and Zaghouani, Wajdi and Li, Chengkai and Shaar, Shaden and Mubarak, Hamdy and Nikolov, Alex and Kartal, Yavuz Selim and Beltr\\'{a}n, Javier},
                title = "Overview of the {CLEF}-2022 {CheckThat}! Lab Task 1 on Identifying Relevant Claims in Tweets",
                year = {2022},
                booktitle = "Working Notes of {CLEF} 2022---Conference and Labs of the Evaluation Forum",
                series = {CLEF~'2022},
                address = {Bologna, Italy},
            }""",
            "link": "https://gitlab.com/checkthat_lab/clef2022-checkthat-lab/clef2022-checkthat-lab",
            "license": "Research Purpose Only",
            "splits": {
                "ar": {
                    "test": "CT22_arabic_1D_attentionworthy_test_gold.tsv",
                    "train": "CT22_arabic_1D_attentionworthy_train.tsv",
                }
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["tweet_text"]
            input_id = row["tweet_id"]
            label = str(row["class_label"]).lower()
            data.append(
                {
                    "input": text,
                    "label": label,
                    "input_id": input_id,
                    "line_number": index,
                }
            )
        return data
