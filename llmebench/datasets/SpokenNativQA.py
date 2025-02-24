import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class SpokenNativQADataset(DatasetBase):
    def __init__(self, **kwargs):
        super(SpokenNativQADataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "data_id": "a unique question id",
            "input": {
                "question": "question to be answered",
                "length": "number of words in answer",
            },
            "label": "A long answer",
        }

    @staticmethod
    def metadata():
        return {
            "language": "multilingual",
            "citation": """
            citation text goes here
            """,
            "link": "",
            "license": "",
            "splits": {
                "arabic_qa_google": {
                    "test": "arabic_qa/spokenqa_arabic_qa_test_google_asr.jsonl",
                },
                "arabic_qa_whisper": {
                    "test": "arabic_qa/spokenqa_arabic_qa_test_whisper_asr.jsonl"
                },
                "arabic_qa_azure": {
                    "test": "arabic_qa/spokenqa_arabic_qa_test_azure_asr.jsonl"
                },
                "english_qa_google": {
                    "test": "",
                },
                "english_qa_whisper": {
                    "test": "",
                },
                "default": [
                    "arabic_qa_google",
                    "arabic_qa_whisper",
                    "arabic_qa_azure",
                    "english_qa_google",
                    "english_qa_whisper",
                ],
            },
            "task_type": TaskType.Other,
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path) as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                obj = json.loads(line)
                id = obj["data_id"]
                question = obj["asr_text"]
                answer = obj["answer"]
                length = len(answer.split())
                data.append(
                    {
                        "data_id": id,
                        "input": {"question": question, "length": length},
                        "label": answer,
                    }
                )
        return data
