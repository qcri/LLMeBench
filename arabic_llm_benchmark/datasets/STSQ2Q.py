import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class Q2QSimDataset(DatasetBase):
    def __init__(self, **kwargs):
        # custom_param_1/2 are passed from `dataset_args` in the benchmark
        # config
        super(Q2QSimDataset, self).__init__(**kwargs)

    def citation(self):
        # This function returns a string with the bib entry for the dataset
        return """
        @inproceedings{seelawi2019nsurl,
            title={NSURL-2019 task 8: Semantic question similarity in arabic},
            author={Seelawi, Haitham and Mustafa, Ahmad and Al-Bataineh, Hesham and Farhan, Wael and Al-Natsheh, Hussein T},
            booktitle={Proceedings of the First International Workshop on NLP Solutions for Under Resourced Languages (NSURL 2019) co-located with ICNLSP 2019-Short Papers},
            pages={1--8},
            year={2019}
        }"""

    def get_data_sample(self):
        return {"input": "السوال الاول السوال الثاني", "label": "1"}

    def load_data(self, data_path):
        # This function loads the data and _must_ return a list of
        # dictionaries, where each dictionary has atleast two keys
        #   "input": this will be sent to the prompt generator
        #   "label": this will be used for evaluation
        # return False
        data = []
        with open(data_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                line = line.rstrip("\r\n").split("\t")
                # Using -index to handle both train and test file with diff # of columns
                data.append({"input": line[-3] + "\t" + line[-2], "label": line[-1]})

        return data
