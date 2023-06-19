import json

from arabic_llm_benchmark.datasets.SQuADBase import SQuADBase


class TyDiQADataset(SQuADBase):
    def __init__(self, **kwargs):
        super(TyDiQADataset, self).__init__(**kwargs)

    def citation(self):
        return """ @article{tydiqa,
                title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
                author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
                year    = {2020},
                journal = {Transactions of the Association for Computational Linguistics}
            } """
