import json

from llmebench.datasets.SQuADBase import SQuADBase


class XQuADDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(XQuADDataset, self).__init__(**kwargs)

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
