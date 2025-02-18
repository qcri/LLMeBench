import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ProppyBinaryDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ProppyBinaryDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "en",
            "citation": """
                to add
            """,
            "link": "",
            "license": "",
            "splits": {
                "test": "/Users/mhasanain/work/temp_propoganda/propaganda_detector/data/english_formatted/binary/EN_sentences_binary_test_no_clef_formatted.jsonl",
                "dev": "/Users/mhasanain/work/temp_propoganda/propaganda_detector/data/english_formatted/binary/EN_sentences_binary_dev_no_clef_formatted.jsonl",
                "train": "/Users/mhasanain/work/temp_propoganda/propaganda_detector/data/english_formatted/binary/EN_sentences_binary_train_no_clef_formatted.jsonl",
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
