from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArSASDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArSASDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{Elmadany2018ArSASA,
                title={ArSAS : An Arabic Speech-Act and Sentiment Corpus of Tweets},
                author={AbdelRahim Elmadany and Hamdy Mubarak and Walid Magdy},
                year={2018}
            }""",
            "link": "https://homepages.inf.ed.ac.uk/wmagdy/resources.htm",
            "license": "Research Purpose Only",
            "splits": {
                "test": "ArSAS-test.txt",
                "train": "ArSAS-train.txt",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["Positive", "Negative", "Neutral", "Mixed"],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Tweet", "label": "Positive"}

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                text, label = line.strip().split("\t")
                data.append({"input": text, "label": label, "line_number": line_idx})

        return data
