import csv

from llmebench.datasets.dataset_base import DatasetBase

from llmebench.tasks import TaskType


class ArSarcasm2Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArSarcasm2Dataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{abufarha-etal-2021-arsarcasm-v2,
                title = "Overview of the WANLP 2021 Shared Task on Sarcasm and Sentiment Detection in Arabic",
                author = "Abu Farha, Ibrahim  and
                Zaghouani, Wajdi  and
                Magdy, Walid",
                booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
                month = april,
                year = "2021",
            }""",
            "link": "https://github.com/iabufarha/ArSarcasm-v2",
            "license": "MIT License",
            "splits": {
                "test": "testing_data.csv",
                "train": "training_data.csv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["TRUE", "FALSE"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "A tweet", "label": "TRUE"}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for line_idx, row in enumerate(reader):
                data.append(
                    {
                        "input": row["tweet"],
                        "label": row[
                            "sarcasm"
                        ].upper(),  # To get it to work on ArSarcasm (True/False) and ArSarcasm-2 (TRUE/FALSE)
                        "line_number": line_idx,
                    }
                )

        return data
