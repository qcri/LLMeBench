import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class FactualityKhouja20Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(FactualityKhouja20Dataset, self).__init__(**kwargs)

    def citation(self):
        return """
        @article{khouja2020stance,
            title={Stance prediction and claim verification: An Arabic perspective},
            author={Khouja, Jude},
            journal={arXiv preprint arXiv:2005.10410},
            year={2020}
        }"""

    def get_data_sample(self):
        return {"input": "الجملة بالعربية", "label": "yes"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []
        raw_data = pd.read_csv(data_path, sep=",")
        for index, row in raw_data.iterrows():
            sentence = row["claim_s"]
            label_fixed = str(row["fake_flag"])
            if label_fixed == "1":
                label_fixed = "true"
            elif label_fixed == "0":
                label_fixed = "false"
            data.append({"input": sentence, "label": label_fixed})
        return data
