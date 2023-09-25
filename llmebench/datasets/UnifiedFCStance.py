import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class UnifiedFCStanceDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(UnifiedFCStanceDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{baly2018integrating,
                title = "Integrating Stance Detection and Fact Checking in a Unified Corpus",
                author = "Baly, Ramy  and
                  Mohtarami, Mitra  and
                  Glass, James  and
                  M{\\`a}rquez, Llu{\\'\\i}s  and
                  Moschitti, Alessandro  and
                  Nakov, Preslav",
                booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
                year = "2018",
            }""",
            "link": "https://alt.qcri.org/resources/arabic-fact-checking-and-stance-detection-corpus/",
            "license": "Research Purpose Only",
            "splits": {
                "test": "ramy_arabic_stance.jsonl",
                "train": ":data_dir:ANSStance/stance/train.csv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["agree", "disagree", "discuss", "unrelated"],
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": {
                # Train samples
                "sentence_1": "sentence 1 text",
                "sentence_2": "sentence 2 text",
                # Test samples
                "claim": "الجملة الاولى",
                "claim-fact": "الجملة الاولى",
                "article": "الجملة الثانية",
            },
            "label": "agree",
        }

    def load_train_data(self, data_path):
        # Training data is used from StanceKhouja as
        # no native training data is available
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r", encoding="utf-8") as fp:
            next(fp)  # skip header
            for line_idx, line in enumerate(fp):
                s1, s2, label = line.strip().split(",")

                data.append(
                    {
                        "input": {"sentence_1": s1.strip(), "sentence_2": s2.strip()},
                        "label": label,
                        "line_number": line_idx,
                    }
                )

        return data

    def load_data(self, data_path, no_labels=False):
        data = []

        if "train" in data_path:
            return self.load_train_data(data_path)
        else:
            data_path = self.resolve_path(data_path)
            with open(data_path, "r", encoding="utf-8") as json_file:
                for line in json_file:
                    json_obj = json.loads(line)

                    data.append(
                        {
                            "input": {
                                "claim": json_obj["claim"],
                                "claim-fact": json_obj["claim-fact"],
                                "article": json_obj["article"],
                            },
                            "label": json_obj["stance"],
                        }
                    )
            return data
