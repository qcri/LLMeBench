import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class NewsCatAlArabiyaDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(NewsCatAlArabiyaDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {"input": "some tweet", "label": "checkworthy"}

    def citation(self):
        return """
                @article{einea2019sanad,
                  title={Sanad: Single-label {A}rabic news articles dataset for automatic text categorization},
                  author={Einea, Omar and Elnagar, Ashraf and Al Debsi, Ridhwan},
                  journal={Data in brief},
                  volume={25},
                  pages={104076},
                  year={2019},
                  publisher={Elsevier}
                }
        """

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        dir_path = "data/news_categorization/"
        for index, row in raw_data.iterrows():
            filename = row["file_path"].strip()
            file = open(dir_path + filename, "r")
            lines = file.readlines()
            lines = " ".join(lines).strip()
            label = row["class_label"]

            entry = {"id": filename, "input": lines, "label": label}
            data.append(entry)

        return data
