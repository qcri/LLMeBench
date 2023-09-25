import json

from llmebench.datasets.SQuADBase import SQuADBase
from llmebench.tasks import TaskType


class TyDiQADataset(SQuADBase):
    def __init__(self, **kwargs):
        super(TyDiQADataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@article{tydiqa,
                title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
                author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
                year    = {2020},
                journal = {Transactions of the Association for Computational Linguistics}
            }""",
            "link": "https://github.com/google-research-datasets/tydiqa",
            "license": "Apache License Version 2.0",
            "splits": {
                "dev": "tydiqa-goldp-dev-arabic.json",
                "train": ":data_dir:ARCD/arcd-train.json",
            },
            "task_type": TaskType.QuestionAnswering,
        }
