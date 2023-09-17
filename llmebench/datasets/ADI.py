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
                "dev": "data/sequence_tagging_ner_pos_etc/dialect_identification/fewshot_dev.tsv",
                "test": "data/sequence_tagging_ner_pos_etc/dialect_identification/all_v2.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": [
                "EGY",
                "IRA",
                "JOR",
                "KSA",
                "KUW",
                "LEB",
                "LIB",
                "MOR",
                "MSA",
                "PAL",
                "QAT",
                "SUD",
                "SYR",
                "UAE",
                "YEM",
            ],
        }

    def load_data(self, data_path):
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
