import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ADIDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ADIDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {"input": "some tweet", "label": "no_not_interesting"}

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """TO DO: in house dataset""",
            "splits": {
                "dev": "fewshot_dev.tsv",
                "test": "all_v2.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "egy",
                "ira",
                "jor",
                "ksa",
                "kuw",
                "leb",
                "lib",
                "mor",
                "msa",
                "pal",
                "qat",
                "sud",
                "syr",
                "uae",
                "YEM",
            ],
        }

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["text"]
            input_id = row["SegId"]
            label = str(row["RefLabel"]).lower()
            data.append(
                {
                    "input": text,
                    "label": label,
                    "input_id": input_id,
                    "line_number": index,
                }
            )
        return data
