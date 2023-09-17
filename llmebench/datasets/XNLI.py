import pandas as pd

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class XNLIDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(XNLIDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@InProceedings{conneau2018xnli,
                author = "Conneau, Alexis
                    and Rinott, Ruty
                    and Lample, Guillaume
                    and Williams, Adina
                    and Bowman, Samuel R.
                    and Schwenk, Holger
                    and Stoyanov, Veselin",
                title = "XNLI: Evaluating Cross-lingual Sentence Representations",
                booktitle = "Proceedings of the 2018 Conference on Empirical Methods
                           in Natural Language Processing",
                year = "2018",
                publisher = "Association for Computational Linguistics",
                location = "Brussels, Belgium",
            }""",
            "link": "https://github.com/facebookresearch/XNLI",
            "license": "CC BY-NC 4.0",
            "splits": {
                "dev": "xnli.dev.tsv",
                "test": "xnli.test.ar.tsv",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["contradiction", "entailment", "neutral"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Test\tTest", "label": "neutral"}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        formatted_data = []

        with open(data_path, "r", encoding="utf-8") as in_file:
            next(in_file)
            for index, line in enumerate(in_file):
                if not line.startswith("ar"):
                    continue
                line = [str(s.strip()) for s in line.rstrip("\r\n").split("\t")]

                sent1_sent2 = line[6] + "\t" + line[7]
                label = line[1]
                pid = line[9]

                formatted_data.append(
                    {
                        "input": sent1_sent2,
                        "label": label,
                        "line_number": index,
                        "input_id": pid,
                    }
                )
        print("loaded %d data samples from file!" % len(formatted_data))
        return formatted_data
