import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class FactualityUnifiedFCDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(FactualityUnifiedFCDataset, self).__init__(**kwargs)

    def get_data_sample(self):
        return {"input": "some tweet", "label": "True"}

    def citation(self):
        return """
                @inproceedings{baly2018integrating,
                  title = "Integrating Stance Detection and Fact Checking in a Unified Corpus",
                    author = "Baly, Ramy  and
                      Mohtarami, Mitra  and
                      Glass, James  and
                      M{\`a}rquez, Llu{\'\i}s  and
                      Moschitti, Alessandro  and
                      Nakov, Preslav",
                    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
                    year = "2018",
                }
        """

    def load_data(self, data_path):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["text"]
            label = str(row["class_label"])
            data.append(
                {
                    "input": text,
                    "label": label,
                    "line_number": index,
                }
            )
        return data
