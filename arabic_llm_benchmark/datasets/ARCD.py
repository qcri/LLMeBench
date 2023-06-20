import json

from arabic_llm_benchmark.datasets.SQuADBase import SQuADBase


class ARCDDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(ARCDDataset, self).__init__(**kwargs)

    def citation(self):
        return """
        @misc{mozannar2019neural,
            title={Neural Arabic Question Answering}, 
            author={Hussein Mozannar and Karl El Hajal and Elie Maamary and Hazem Hajj},
            year={2019},
            eprint={1906.05394},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }"""
