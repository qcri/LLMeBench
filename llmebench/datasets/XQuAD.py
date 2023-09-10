import json

from llmebench.datasets.SQuADBase import SQuADBase


class XQuADDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(XQuADDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """
            @inproceedings{artetxe2020cross,
              title={On the Cross-lingual Transferability of Monolingual Representations},
              author={Artetxe, Mikel and Ruder, Sebastian and Yogatama, Dani},
              booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
              pages={4623--4637},
              year={2020}
            }
            """,
        }
