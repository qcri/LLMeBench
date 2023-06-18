import json

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class FactualityUnifiedFCDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(FactualityUnifiedFCDataset, self).__init__(**kwargs)

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
            "input": "الجملة الاولى",
            "label": "agree",
            "input_id": "id"
        }

    def load_data(self, data_path):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            next(f)
            for line_idx, line in enumerate(f):
                input_id, sentence, label = [str(s.strip()) for s in line.split("\t")]

                data.append({"input": sentence, "label": label, "line_number": line_idx, "input_id": input_id})

        return data
