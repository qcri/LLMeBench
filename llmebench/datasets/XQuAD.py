import json

from llmebench.datasets.SQuADBase import SQuADBase
from llmebench.tasks import TaskType


class XQuADDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(XQuADDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{artetxe2020cross,
                title={On the Cross-lingual Transferability of Monolingual Representations},
                author={Artetxe, Mikel and Ruder, Sebastian and Yogatama, Dani},
                booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
                pages={4623--4637},
                year={2020}
            }""",
            "link": "https://github.com/google-deepmind/xquad",
            "license": "CC-BY-SA4.0",
            "splits": {
                "test": "xquad.ar.json",
                "train": ":data_dir:ARCD/arcd-train.json",
            },
            "task_type": TaskType.QuestionAnswering,
        }
