from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class WikiNewsSegmentationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(WikiNewsSegmentationDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{darwish2016farasa,
                title={Farasa: A new fast and accurate {A}rabic word segmenter},
                author={Darwish, Kareem and Mubarak, Hamdy},
                booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
                pages={1070--1074},
                year={2016}
            }""",
            "link": "https://github.com/kdarwish/Farasa/blob/master/WikiNews.pos.ref",
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
            "label": "Sentence with segmented words",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)

        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                data.append(
                    {
                        "input": line.replace("+", "").strip(),
                        "label": line.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
