import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class SubjectivityDataset(DatasetBase):
    def __init__(self):
        pass

    def load_data(self, data_path, no_labels=False):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["sentence"]
            label = str(row["label"])
            data.append({"input": text, "label": label, "line_number": index})
        return data
