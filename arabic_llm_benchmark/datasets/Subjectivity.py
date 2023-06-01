import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class SubjectivityDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(SubjectivityDataset, self).__init__(**kwargs)

    def citation(self):
        return """
                @inproceedings{barron2023clef,
                  title={The CLEF-2023 CheckThat! Lab: Checkworthiness, Subjectivity, Political Bias, Factuality, and Authority},
                  author={Barr{\'o}n-Cede{\~n}o, Alberto and Alam, Firoj and Caselli, Tommaso and Da San Martino, Giovanni and Elsayed, Tamer and Galassi, Andrea and Haouari, Fatima and Ruggeri, Federico and Stru{\ss}, Julia Maria and Nandi, Rabindra Nath and others},
                  booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part III},
                  pages={506--517},
                  year={2023},
                  organization={Springer}
                }
            """

    def load_data(self, data_path, no_labels=False):
        data = []
        raw_data = pd.read_csv(data_path, sep="\t")
        for index, row in raw_data.iterrows():
            text = row["sentence"]
            label = str(row["label"])
            data.append({"input": text, "label": label, "line_number": index})
        return data
