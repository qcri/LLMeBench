from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class WikiNewsDiacritizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(WikiNewsDiacritizationDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{darwish-etal-2017-arabic,
                title = "{A}rabic Diacritization: Stats, Rules, and Hacks",
                author = "Darwish, Kareem  and
                  Mubarak, Hamdy  and
                  Abdelali, Ahmed",
                booktitle = "Proceedings of the Third {A}rabic Natural Language Processing Workshop",
                month = apr,
                year = "2017",
                address = "Valencia, Spain",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/W17-1302",
                doi = "10.18653/v1/W17-1302",
                pages = "9--17",
            }""",
            "link": "https://github.com/kdarwish/Farasa/tree/master",
            "license": "Research Purpose Only",
            "splits": {
                "test": "WikiNewsTruth.txt",
                "train": "WikiNewsTruthDev.txt",
            },
            "task_type": TaskType.Other,
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": "Original sentence",
            "label": "Sentence with diacritized words",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                text, diacritized_text = line.split("\t")
                data.append(
                    {
                        "input": text.strip(),
                        "label": diacritized_text.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
