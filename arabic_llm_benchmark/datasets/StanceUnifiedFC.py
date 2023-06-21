import json
import os
import pandas as pd

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class StanceUnifiedFCDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(StanceUnifiedFCDataset, self).__init__(**kwargs)

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

    def get_data_sample(self):
        return {
            "input": {
                "claim": "الجملة الاولى",
                "claim-fact": "الجملة الاولى",
                "article": "الجملة الثانية",
            },
            "label": "agree",
        }

    def load_train_data(self, data_path):
        # TODO: modify to iterator
        data = []
        with open(data_path, "r", encoding="utf-8") as fp:
            next(fp)  # skip header
            for line_idx, line in enumerate(fp):
                s1, s2, label = line.strip().split(",")

                # Had to concatenate s1 and s2 this as langchain only accepts strings
                data.append(
                        {"input": s1.strip() + "\t" + s2.strip(), "label": label, "line_number": line_idx}
                )

        return data


    def load_data(self, data_path, no_labels=False):
        data = []

        if "train" in data_path:
            return self.load_train_data(data_path)
        else:
            with open(data_path, "r", encoding="utf-8") as json_file:
                for line in json_file:
                    json_obj = json.loads(line)
                    #Had to make input a string instead of a dictionar{claim,article} to get FS to work
                    data.append(
                            {"input": "claim: " + str(json_obj["claim"])+ "\tarticle: " + str(json_obj["article"]),
                             "label": json_obj["stance"]})
            return data
