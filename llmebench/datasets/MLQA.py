import json

from llmebench.datasets.SQuADBase import SQuADBase


class MLQADataset(SQuADBase):
    def __init__(self, **kwargs):
        super(MLQADataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@article{lewis2019mlqa,
                title=MLQA: Evaluating Cross-lingual Extractive Question Answering,
                author={Lewis, Patrick and Ouguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
                journal={arXiv preprint arXiv:1910.07475},
                year={2019}
            }""",
        }
