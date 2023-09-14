from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class WikiNewsPOSDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(WikiNewsPOSDataset, self).__init__(**kwargs)

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
                "test": "data/sequence_tagging_ner_pos_etc/POS/WikiNewsTruth.txt.POS.tab",
                "train": "data/sequence_tagging_ner_pos_etc/POS/WikiNewsTruthDev.txt",
            },
            "task_type": TaskType.Labeling,
            "class_labels": [],
        }

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with POS tags",
        }

    def load_data(self, data_path, no_labels=False):
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
