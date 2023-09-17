from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class WikiNewsPOSDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(WikiNewsPOSDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{darwish2017arabic,
                title={Arabic {POS} tagging: Don’t abandon feature engineering just yet},
                author={Darwish, Kareem and Mubarak, Hamdy and Abdelali, Ahmed and Eldesouki, Mohamed},
                booktitle={Proceedings of the third arabic natural language processing workshop},
                pages={130--137},
                year={2017}
            }""",
            "link": "https://github.com/kdarwish/Farasa/blob/master/WikiNews.pos.ref",
            "license": "Research Purpose Only",
            "splits": {
                "test": "WikiNewsTruth.txt.POS.tab",
                "train": "WikiNewsTruthDev.txt",
            },
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "ABBREV",
                "ADJ",
                "ADJ/CONJ",
                "ADJ/DET",
                "ADJ/NUM",
                "ADV",
                "CASE",
                "CONJ",
                "DET",
                "FOREIGN",
                "FUT_PART",
                "NOUN",
                "NOUN/DET",
                "NSUFF",
                "NSUFF/ADJ",
                "NSUFF/DET",
                "NSUFF/NOUN",
                "NUM",
                "PART",
                "PART/CONJ",
                "PART/NOUN",
                "PART/PART",
                "PART/PREP",
                "PREP",
                "PRON",
                "PUNC",
                "V",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": "Original sentence",
            "label": "Sentence with POS tags",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                data.append(
                    {
                        "input": line.strip().split("\t")[0],
                        "label": line.strip().split("\t")[1],
                        "line_number": line_idx,
                    }
                )

        return data
