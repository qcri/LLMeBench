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

    def load_data(self, data_path, no_labels=False):
        data = []

        data_path, train_or_test, fold_id  = data_path.split('\t')
        dir_name = os.path.dirname(data_path)
        fold_path = dir_name + '/train_test_splits.txt'
        fold_data = pd.read_csv(fold_path, engine='python', delim_whitespace=True, header=None)
        fold_data[0] = fold_data[0].str.strip(':')
        fold_data = fold_data.transpose()
        fold_data.columns = fold_data.loc[0, :].to_list()
        fold_data.drop(0, inplace=True)
        if train_or_test =='train':
            selected_ids = fold_data.drop(fold_id, axis=1).values.flatten().tolist()
        elif train_or_test =='test':
            selected_ids = fold_data[fold_id].to_list()
        

        with open(data_path, "r") as json_file:
            for line in json_file:
                json_obj = json.loads(line)
                
                if json_obj['id'] in selected_ids:
                    data.append(
                        {
                            "input": json_obj,
                            "label": json_obj["stance"],
                        }
                    )
        return data
