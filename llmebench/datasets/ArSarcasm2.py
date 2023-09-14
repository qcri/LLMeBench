import csv

from llmebench.datasets.dataset_base import DatasetBase


class ArSarcasm2Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArSarcasm2Dataset, self).__init__(**kwargs)

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
        }

    def get_data_sample(self):
        return {"input": "A tweet", "label": "TRUE"}

    def load_data(self, data_path):
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
