import json

from arabic_llm_benchmark.datasets.SQuADBase import SQuADBase


class MlqaDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(MlqaDataset, self).__init__(**kwargs)

    def citation(self):
        return """ @article{lewis2019mlqa,
        title=MLQA: Evaluating Cross-lingual Extractive Question Answering,
        author={Lewis, Patrick and Ouguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
        journal={arXiv preprint arXiv:1910.07475},
        year={2019} """
