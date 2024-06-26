import csv

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class MultiNativQADataset(DatasetBase):
    def __init__(self, **kwargs):
        super(MultiNativQADataset, self).__init__(**kwargs)

    @staticmethod
    def get_data_sample():
        return {
            "input": {
                "question": "question to be answered",
                "question_id": "a unique question id",
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
                "arabic_qa": {
                    "dev": "arabic_qa/NativQA_ar_msa_qa_dev.tsv",
                    "test": "arabic_qa/NativQA_ar_msa_qa_test.tsv",
                },
                "assamese_in": {
                    "dev": "assamese_in/NativQA_asm_NA_as_dev.tsv",
                    "test": "assamese_in/NativQA_asm_NA_as_test.tsv",
                },
                "bangla_bd": {
                    "dev": "bangla_bd/NativQA_bn_scb_bd_dev.tsv",
                    "test": "bangla_bd/NativQA_bn_scb_bd_test.tsv",
                },
                "bangla_in": {
                    "dev": "bangla_in/NativQA_bn_scb_in_dev.tsv",
                    "test": "bangla_in/NativQA_bn_scb_in_test.tsv",
                },
                "english_bd": {
                    "dev": "english_bd/NativQA_en_NA_bd_dev.tsv",
                    "test": "english_bd/NativQA_en_NA_bd_test.tsv",
                },
                "english_qa": {
                    "dev": "english_qa/NativQA_en_NA_qa_dev.tsv",
                    "test": "english_qa/NativQA_en_NA_qa_test.tsv",
                },
                "hindi_in": {
                    "dev": "hindi_in/NativQA_hi_NA_in_dev.tsv",
                    "test": "hindi_in/NativQA_hi_NA_in_test.tsv",
                },
                "nepali_np": {
                    "dev": "nepali_np/NativQA_ne_NA_np_dev.tsv",
                    "test": "nepali_np/NativQA_ne_NA_np_test.tsv",
                },
                "turkish_tr": {
                    "dev": "english_qa/NativQA_tr_NA_tr_dev.tsv",
                    "test": "english_qa/NativQA_tr_NA_tr_test.tsv",
                },
                "default": [
                    "assamese_in",
                    "bangla_bd",
                    "bangla_in",
                    "english_bd",
                    "english_bd",
                    "hindi_in",
                    "nepali_np",
                    "english_qa",
                ],
                # "default": ["arabic_qa"],
            },
            "task_type": TaskType.Other,
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                id = row[0]
                question = row[3]
                answer = row[4]
                data.append(
                    {
                        "data_id": id,
                        "input": question,
                        "label": answer,
                    }
                )
        return data
