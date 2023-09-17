import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class CT23SubjectivityDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CT23SubjectivityDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {"input": "some tweet", "label": "SUBJ"}

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{barron2023clef,
                title={The CLEF-2023 CheckThat! Lab: Checkworthiness, Subjectivity, Political Bias, Factuality, and Authority},
                author={Barr{\\'o}n-Cede{\\~n}o, Alberto and Alam, Firoj and Caselli, Tommaso and Da San Martino, Giovanni and Elsayed, Tamer and Galassi, Andrea and Haouari, Fatima and Ruggeri, Federico and Stru{\\ss}, Julia Maria and Nandi, Rabindra Nath and others},
                booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part III},
                pages={506--517},
                year={2023},
                organization={Springer}
            }""",
            "link": "https://gitlab.com/checkthat_lab/clef2023-checkthat-lab",
            "license": "CC BY NC SA 4.0",
            "splits": {
                "ar": {
                    "dev": "dev_ar.tsv",
                    "train": "train_ar.tsv",
                }
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
