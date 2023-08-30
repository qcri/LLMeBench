import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase


class FactualityKhouja20Dataset(DatasetBase):
    def __init__(self, **kwargs):
        super(FactualityKhouja20Dataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@article{khouja2020stance,
                title={Stance prediction and claim verification: An Arabic perspective},
                author={Khouja, Jude},
                journal={arXiv preprint arXiv:2005.10410},
                year={2020}
            }""",
        }

    def get_data_sample(self):
        return {"input": "الجملة بالعربية", "label": "true", "line_number": "1"}

    def load_data(self, data_path, no_labels=False):
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            next(f)
            for line_idx, line in enumerate(f):
                sentence, label_fixed = [str(s.strip()) for s in line.split(",")]

                # The dataset uses 1 to reflect false/fake claims
                if label_fixed == "1":
                    label_fixed = "false"
                elif label_fixed == "0":
                    label_fixed = "true"

                data.append(
                    {"input": sentence, "label": label_fixed, "line_number": line_idx}
                )

        return data
