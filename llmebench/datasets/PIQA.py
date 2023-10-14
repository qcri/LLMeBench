from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType
import pandas as pd
import json


class PIQADataset(DatasetBase):
    def __init__(self, **kwargs):
        super(PIQADataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """@inproceedings{bisk2020piqa,
                    title={Piqa: Reasoning about physical commonsense in natural language},
                    author={Bisk, Yonatan and Zellers, Rowan and Gao, Jianfeng and Choi, Yejin and others},
                    booktitle={Proceedings of the AAAI conference on artificial intelligence},
                    volume={34},
                    number={05},
                    pages={7432--7439},
                    year={2020}
                    }
                """,
            "link": "https://huggingface.co/datasets/piqa",
            "download_url": "https://yonatanbisk.com/piqa/data/",
            "splits": {
                "train": "train",
                "dev": "dev",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["0", "1"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "When boiling butter, when it's ready, you can", "sol1": "Pour it onto a plate", "sol2": "Pour it into a jar", "label": "1"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path + ".jsonl")
        label_path = self.resolve_path(data_path + "-labels.lst")
        data = []
        label_data = pd.read_csv(label_path, sep="\t", header=None)
        
        with open(data_path, "r", encoding="utf-8") as json_file:
                for index, line in enumerate(json_file):
                    json_obj = json.loads(line)
                    label = label_data.loc[index]
                    data.append(
                    {"input": json_obj["goal"], "sol1": json_obj["sol1"], "sol2":json_obj["sol2"], "label": label}
                    )
        return data
