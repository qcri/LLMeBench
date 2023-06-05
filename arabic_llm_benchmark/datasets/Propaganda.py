import json

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class PropagandaTweetDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(PropagandaTweetDataset, self).__init__(**kwargs)

    def citation(self):
        return """@article{wanlp2023,
          year={2023}
        }"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": ["no technique"]}

    def load_data(self, data_path):
        data = []
        with open(data_path, mode='r', encoding="utf-8") as infile:
            json_data = json.load(infile)
            for index, tweet in enumerate(json_data):
                text = tweet["text"]
                label = tweet["labels"]
                data.append({"input": text, "label": label, "line_number": index})

        return data
