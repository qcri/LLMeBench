import json

import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArabicLJPDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArabicLJPDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
                @misc{kmainasi2025largelanguagemodelspredict,
                title={Can Large Language Models Predict the Outcome of Judicial Decisions?}, 
                author={Mohamed Bayan Kmainasi and Ali Ezzat Shahroor and Amani Al-Ghraibah},
                year={2025},
                eprint={2501.09768},
                archivePrefix={arXiv},
                primaryClass={cs.CL},
                url={https://arxiv.org/abs/2501.09768}, 
            }                
            }""",
            "link": "https://huggingface.co/datasets/mbayan/Arabic-LJP",
            "license": "CC BY-NC 4.0",
            "splits": {
                "train": "train.json",
                "test": "test.json",
            },
            "task_type": TaskType.NLGenerationTask,
        }

    @staticmethod
    def get_data_sample():
        return {
            "id": "a unique id",
            "input": {
                "input": "A legal case.",
                "instruction": "instruction in Arabic",
            },
            "label": "Legal judgement outcome.",
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        # Load JSON data into a pandas DataFrame
        df = pd.read_json(data_path, encoding="utf-8")

        # Format the data as required
        df["formatted_data"] = df.apply(
            lambda row: {
                "input": {
                    "input": row["input"],
                    "instruction": row["Instruction"],
                },
                "label": row["output"],
                "line_number": row["id"],
            },
            axis=1,
        )

        formatted_data = df["formatted_data"].tolist()

        print(f"Loaded {len(formatted_data)} data samples from file!")
        return formatted_data
