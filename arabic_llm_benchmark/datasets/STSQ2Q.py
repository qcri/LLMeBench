import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class Q2QSimDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(Q2QSimDataset, self).__init__(**kwargs)

    def citation(self):
        return """
        @inproceedings{seelawi2019nsurl,
            title={NSURL-2019 task 8: Semantic question similarity in arabic},
            author={Seelawi, Haitham and Mustafa, Ahmad and Al-Bataineh, Hesham and Farhan, Wael and Al-Natsheh, Hussein T},
            booktitle={Proceedings of the First International Workshop on NLP Solutions for Under Resourced Languages (NSURL 2019) co-located with ICNLSP 2019-Short Papers},
            pages={1--8},
            year={2019}
        }"""

    def get_data_sample(self):
        return {
            "input": "السوال الاول السوال الثاني\tالسوال الاول السوال الثاني",
            "label": "1",
        }

    def load_data(self, data_path):
        data = []
        with open(data_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                line = line.rstrip("\r\n").split("\t")
                # Using -index to handle both train and test file with diff # of columns
                data.append({"input": line[-3] + "\t" + line[-2], "label": line[-1]})

        return data
