import base64
import json
import os

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class SpokenNativQAAudioDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(SpokenNativQAAudioDataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "data_id": "a unique question id",
            "input": {
                "question": "question to be answered",
                "length": "number of words in answer",
                "wav": "base64",
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

    # Function to encode the audio
    def encode_wav(self, wav_path):
        with open(wav_path, "rb") as wav_file:
            wav_data = wav_file.read()
            return base64.b64encode(wav_data).decode("utf-8")

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path) as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                obj = json.loads(line)
                id = obj["data_id"]
                question = obj["asr_text"]
                base_dir = os.path.dirname(data_path)
                file_path = base_dir + "/" + obj["file_path"]
                answer = obj["answer"]
                length = len(answer.split())
                base64_wav = self.encode_wav(file_path)
                data.append(
                    {
                        "input": {
                            "question": question,
                            "length": length,
                            "wav": base64_wav,
                        },
                        "label": answer,
                        "line_number": id,
                    }
                )
        return data
