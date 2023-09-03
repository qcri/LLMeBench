import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase


class DialectADIDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(DialectADIDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {"input": "some tweet", "label": "no_not_interesting"}

    def metadata():
        return {
            "language": "ar",
            "citation": """TO DO: in house dataset""",
        }

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["text"]
            input_id = row["SegId"]
            label = str(row["RefLabel"]).lower()
            data.append(
                {
                    "input": text,
                    "label": label,
                    "input_id": input_id,
                    "line_number": index,
                }
            )
        return data
