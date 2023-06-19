import json

from arabic_llm_benchmark.datasets.SQuADBase import SQuADBase


class TydiQADataset(SQuADBase):
    def __init__(self, **kwargs):
        super(TydiQADataset, self).__init__(**kwargs)

    def citation(self):
        return """ @article{tydiqa,
                title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
                author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
                year    = {2020},
                journal = {Transactions of the Association for Computational Linguistics}
            } """
