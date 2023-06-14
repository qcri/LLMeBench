import csv

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class ArSarcasmDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArSarcasmDataset, self).__init__(**kwargs)

    def citation(self):
        return """...."""

    def get_data_sample(self):
        return {"input": "A tweet", "label": "TRUE"}

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r') as fp:
            reader = csv.DictReader(fp)
            for line_idx, row in enumerate(reader):
                data.append({"input": row['tweet'], "label": row['sarcasm'],  "line_number": line_idx})

        return data
