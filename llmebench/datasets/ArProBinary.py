import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArProBinaryDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArProBinaryDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
                to add
            """,
            "link": "",
            "license": "",
            "splits": {
                "test": "/Users/mhasanain/work/temp_propoganda/data_annotation/final_full_annot/final_split/binary/ArMPro_binary_test_V2.jsonl",
                "dev": "/Users/mhasanain/work/temp_propoganda/data_annotation/final_full_annot/final_split/binary/final_full_annot_clean_binary_dev.jsonl",
                "train": "/Users/mhasanain/work/temp_propoganda/data_annotation/final_full_annot/final_split/binary/final_full_annot_clean_binary_train.jsonl",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["true", "false"],
        }

    @staticmethod
    def get_data_sample():
        return {"id": "001", "input": "paragraph", "label": "true", "type": "paragraph"}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                line_data = json.loads(line)
                id = line_data.get("paragraph_id") or line_data.get("tweet_id")
                text = line_data.get("paragraph") or line_data.get("text")
                label = line_data.get("label", "").lower()
                data.append({"input": text, "label": label, "line_number": id})

        return data
