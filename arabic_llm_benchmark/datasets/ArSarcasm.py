import csv

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class ArSarcasmDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArSarcasmDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{abu-farha-magdy-2020-arabic,
            title = "From {A}rabic Sentiment Analysis to Sarcasm Detection: The {A}r{S}arcasm Dataset",
            author = "Abu Farha, Ibrahim  and Magdy, Walid",
            booktitle = "Proceedings of the 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection",
            month = may,
            year = "2020",
            address = "Marseille, France",
            publisher = "European Language Resource Association",
            url = "https://www.aclweb.org/anthology/2020.osact-1.5",
            pages = "32--39",
            language = "English",
            ISBN = "979-10-95546-51-1",
        }"""

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
