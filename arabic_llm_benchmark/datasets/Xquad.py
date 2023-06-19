import json

from arabic_llm_benchmark.datasets.SQuADBase import SQuADBase


class XquadDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(XquadDataset, self).__init__(**kwargs)

    def citation(self):
        return """@article{Artetxe:etal:2019,
                    author={Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
                    title={On the cross-lingual transferability of monolingual representations},
                    journal={CoRR},
                    volume={abs/1910.11856},
                    year={2019},
                    archivePrefix={arXiv},
                    eprint={1910.11856}
            }"""
