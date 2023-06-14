import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class CovidClaimDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(CovidClaimDataset, self).__init__(**kwargs)

    def citation(self):
        return """@inproceedings{nakov2022overview,
                    title={Overview of the CLEF-2022 CheckThat! lab task 1 on identifying relevant claims in tweets},
                    author={Nakov, Preslav and Barr{\\o}n-Cede{\\~n}o, Alberto and Da San Martino, Giovanni and Alam, Firoj and Kutlu, Mucahid and Zaghouani, Wajdi and Li, Chengkai and Shaar, Shaden and Mubarak, Hamdy and Nikolov, Alex},
                     year={2022},
                    booktitle={Proceedings of the Working Notes of CLEF 2022 - Conference and Labs of the Evaluation Forum}
                }"""

    def get_data_sample(self):
        return {"input": "Tweet", "label": "1"}

    def load_data(self, data_path):
        formatted_data = []

        with open(data_path, "r", encoding="utf-8") as in_file:
            next(in_file)
            for index, line in enumerate(in_file):
                tweet = [str(s.strip()) for s in line.split("\t")]

                text = tweet[3]
                label = tweet[4]
                twt_id = tweet[1]

                formatted_data.append(
                    {
                        "input": text,
                        "label": label,
                        "line_number": index,
                        "input_id": twt_id,
                    }
                )

        return formatted_data
