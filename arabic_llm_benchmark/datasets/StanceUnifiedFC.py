import json

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

        with open(data_path, "r") as json_file:
            for line in json_file:
                json_obj = json.loads(line)
                data.append(
                    {
                        "input": json_obj,
                        "label": json_obj["stance"],
                    }
                )

        return data
