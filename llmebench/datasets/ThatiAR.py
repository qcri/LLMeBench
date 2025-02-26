import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ThatiARDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ThatiARDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {"input": "sentence", "label": "SUBJ"}

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@article{ThatiAR2024,
            title={{ThatiAR}: Subjectivity Detection in Arabic News Sentences},
            author={Reem Suwaileh and Maram Hasanain and Fatema Hubail and Wajdi Zaghouani and Firoj Alam},
            year={2024},
            eprint={2406.05559},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
            }
            """,
            "link": "",
            "license": "CC BY NC SA 4.0",
            "splits": {
                "train": "subjectivity_2024_train.tsv",
                "dev": "subjectivity_2024_dev.tsv",
                "test": "subjectivity_2024_test.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["SUBJ", "OBJ"],
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["sentence"]
            id = row["sentence_id"]
            label = str(row["label"])
            data.append(
                {"input": text, "label": label, "input_id": id, "line_number": index}
            )
        return data
