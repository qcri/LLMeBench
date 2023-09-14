from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class AraBenchDataset(DatasetBase):
    def __init__(self, src, tgt, **kwargs):
        super(AraBenchDataset, self).__init__(**kwargs)
        self.src = src
        self.tgt = tgt

    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{sajjad-etal-2020-arabench,
                title = "{A}ra{B}ench: Benchmarking Dialectal {A}rabic-{E}nglish Machine Translation",
                author = "Sajjad, Hassan  and
                  Abdelali, Ahmed  and
                  Durrani, Nadir  and
                  Dalvi, Fahim",
                booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
                month = dec,
                year = "2020",
                address = "Barcelona, Spain (Online)",
                publisher = "International Committee on Computational Linguistics",
                url = "https://aclanthology.org/2020.coling-main.447",
                doi = "10.18653/v1/2020.coling-main.447",
                pages = "5094--5107"
            }""",
            "link": "https://alt.qcri.org/resources1/mt/arabench/",
            "license": "Apache License, Version 2.0",
            "splits": {
                "APT-LEV": {
                    "dev": (
                        "data/MT/ldc_web_lv.dev.lev.0.lv.ar",
                        "data/MT/ldc_web_lv.dev.lev.0.lv.en",
                    ),
                    "test": (
                        "data/MT/ldc_web_lv.test.lev.0.lv.ar",
                        "data/MT/ldc_web_lv.test.lev.0.lv.en",
                    ),
                    "train": (
                        "data/MT/ldc_web_lv.train.lev.0.lv.ar",
                        "data/MT/ldc_web_lv.train.lev.0.lv.en",
                    ),
                },
                "APT-Nile": {
                    "dev": (
                        "data/MT/ldc_web_eg.dev.nil.0.eg.ar",
                        "data/MT/ldc_web_eg.dev.nil.0.eg.en",
                    ),
                    "test": (
                        "data/MT/ldc_web_eg.test.nil.0.eg.ar",
                        "data/MT/ldc_web_eg.test.nil.0.eg.en",
                    ),
                    "train": (
                        "data/MT/ldc_web_eg.train.nil.0.eg.ar",
                        "data/MT/ldc_web_eg.train.nil.0.eg.en",
                    ),
                },
                "Bible-MGR": {
                    "dev": [
                        (
                            "data/MT/bible.dev.mgr.0.ma.ar",
                            "data/MT/bible.dev.mgr.0.ma.en",
                        ),
                        (
                            "data/MT/bible.dev.mgr.0.tn.ar",
                            "data/MT/bible.dev.mgr.0.tn.en",
                        ),
                    ],
                    "test": [
                        (
                            "data/MT/bible.test.mgr.0.ma.ar",
                            "data/MT/bible.test.mgr.0.ma.en",
                        ),
                        (
                            "data/MT/bible.test.mgr.0.tn.ar",
                            "data/MT/bible.test.mgr.0.tn.en",
                        ),
                    ],
                    "train": [
                        (
                            "data/MT/bible.train.mgr.0.ma.ar",
                            "data/MT/bible.train.mgr.0.ma.en",
                        ),
                        (
                            "data/MT/bible.train.mgr.0.tn.ar",
                            "data/MT/bible.train.mgr.0.tn.en",
                        ),
                    ],
                },
                "Bible-MSA": {
                    "dev": [
                        (
                            "data/MT/bible.dev.msa.0.ms.ar",
                            "data/MT/bible.dev.msa.0.ms.en",
                        ),
                        (
                            "data/MT/bible.dev.msa.1.ms.ar",
                            "data/MT/bible.dev.msa.1.ms.en",
                        ),
                    ],
                    "test": [
                        (
                            "data/MT/bible.test.msa.0.ms.ar",
                            "data/MT/bible.test.msa.0.ms.en",
                        ),
                        (
                            "data/MT/bible.test.msa.1.ms.ar",
                            "data/MT/bible.test.msa.1.ms.en",
                        ),
                    ],
                    "train": [
                        (
                            "data/MT/bible.train.msa.0.ms.ar",
                            "data/MT/bible.train.msa.0.ms.en",
                        ),
                        (
                            "data/MT/bible.train.msa.1.ms.ar",
                            "data/MT/bible.train.msa.1.ms.en",
                        ),
                    ],
                },
                "MADAR-Gulf": {
                    "dev": (
                        "data/MT/madar.dev.glf.0.qa.ar",
                        "data/MT/madar.dev.glf.0.qa.en",
                    ),
                    "test": [
                        (
                            "data/MT/madar.test.glf.0.iq.ar",
                            "data/MT/madar.test.glf.0.iq.en",
                        ),
                        (
                            "data/MT/madar.test.glf.1.iq.ar",
                            "data/MT/madar.test.glf.1.iq.en",
                        ),
                        (
                            "data/MT/madar.test.glf.2.iq.ar",
                            "data/MT/madar.test.glf.2.iq.en",
                        ),
                        (
                            "data/MT/madar.test.glf.0.om.ar",
                            "data/MT/madar.test.glf.0.om.en",
                        ),
                        (
                            "data/MT/madar.test.glf.0.qa.ar",
                            "data/MT/madar.test.glf.0.qa.en",
                        ),
                        (
                            "data/MT/madar.test.glf.0.sa.ar",
                            "data/MT/madar.test.glf.0.sa.en",
                        ),
                        (
                            "data/MT/madar.test.glf.1.sa.ar",
                            "data/MT/madar.test.glf.1.sa.en",
                        ),
                        (
                            "data/MT/madar.test.glf.0.ye.ar",
                            "data/MT/madar.test.glf.0.ye.en",
                        ),
                    ],
                    "train": (
                        "data/MT/madar.train.glf.0.qa.ar",
                        "data/MT/madar.train.glf.0.qa.en",
                    ),
                },
                "MADAR-LEV": {
                    "dev": (
                        "data/MT/madar.dev.lev.0.lb.ar",
                        "data/MT/madar.dev.lev.0.lb.en",
                    ),
                    "test": [
                        (
                            "data/MT/madar.test.lev.0.jo.ar",
                            "data/MT/madar.test.lev.0.jo.en",
                        ),
                        (
                            "data/MT/madar.test.lev.1.jo.ar",
                            "data/MT/madar.test.lev.1.jo.en",
                        ),
                        (
                            "data/MT/madar.test.lev.0.lb.ar",
                            "data/MT/madar.test.lev.0.lb.en",
                        ),
                        (
                            "data/MT/madar.test.lev.0.pa.ar",
                            "data/MT/madar.test.lev.0.pa.en",
                        ),
                        (
                            "data/MT/madar.test.lev.0.sy.ar",
                            "data/MT/madar.test.lev.0.sy.en",
                        ),
                        (
                            "data/MT/madar.test.lev.1.sy.ar",
                            "data/MT/madar.test.lev.1.sy.en",
                        ),
                    ],
                    "train": (
                        "data/MT/madar.train.lev.0.lb.ar",
                        "data/MT/madar.train.lev.0.lb.en",
                    ),
                },
                "MADAR-MGR": {
                    "dev": (
                        "data/MT/madar.dev.mgr.0.ma.ar",
                        "data/MT/madar.dev.mgr.0.ma.en",
                    ),
                    "test": [
                        (
                            "data/MT/madar.test.mgr.0.dz.ar",
                            "data/MT/madar.test.mgr.0.dz.en",
                        ),
                        (
                            "data/MT/madar.test.mgr.0.ly.ar",
                            "data/MT/madar.test.mgr.0.ly.en",
                        ),
                        (
                            "data/MT/madar.test.mgr.1.ly.ar",
                            "data/MT/madar.test.mgr.1.ly.en",
                        ),
                        (
                            "data/MT/madar.test.mgr.0.ma.ar",
                            "data/MT/madar.test.mgr.0.ma.en",
                        ),
                        (
                            "data/MT/madar.test.mgr.1.ma.ar",
                            "data/MT/madar.test.mgr.1.ma.en",
                        ),
                        (
                            "data/MT/madar.test.mgr.0.tn.ar",
                            "data/MT/madar.test.mgr.0.tn.en",
                        ),
                        (
                            "data/MT/madar.test.mgr.1.tn.ar",
                            "data/MT/madar.test.mgr.1.tn.en",
                        ),
                    ],
                    "train": [
                        (
                            "data/MT/madar.train.mgr.0.ma.ar",
                            "data/MT/madar.train.mgr.0.ma.en",
                        ),
                        (
                            "data/MT/madar.train.mgr.0.tn.ar",
                            "data/MT/madar.train.mgr.0.tn.en",
                        ),
                    ],
                },
                "MADAR-MSA": {
                    "dev": (
                        "data/MT/madar.dev.msa.0.ms.ar",
                        "data/MT/madar.dev.msa.0.ms.en",
                    ),
                    "test": (
                        "data/MT/madar.test.msa.0.ms.ar",
                        "data/MT/madar.test.msa.0.ms.en",
                    ),
                    "train": (
                        "data/MT/madar.train.msa.0.ms.ar",
                        "data/MT/madar.train.msa.0.ms.en",
                    ),
                },
                "MADAR-Nile": {
                    "dev": (
                        "data/MT/madar.dev.nil.0.eg.ar",
                        "data/MT/madar.dev.nil.0.eg.en",
                    ),
                    "test": [
                        (
                            "data/MT/madar.test.nil.0.eg.ar",
                            "data/MT/madar.test.nil.0.eg.en",
                        ),
                        (
                            "data/MT/madar.test.nil.1.eg.ar",
                            "data/MT/madar.test.nil.1.eg.en",
                        ),
                        (
                            "data/MT/madar.test.nil.2.eg.ar",
                            "data/MT/madar.test.nil.2.eg.en",
                        ),
                        (
                            "data/MT/madar.test.nil.0.sd.ar",
                            "data/MT/madar.test.nil.0.sd.en",
                        ),
                    ],
                    "train": (
                        "data/MT/madar.train.nil.0.eg.ar",
                        "data/MT/madar.train.nil.0.eg.en",
                    ),
                },
                "MDC-LEV": {
                    "dev": (
                        "data/MT/ldc_web_eg.dev.lev.0.sy.ar",
                        "data/MT/ldc_web_eg.dev.lev.0.sy.en",
                    ),
                    "test": [
                        (
                            "data/MT/ldc_web_eg.test.lev.0.jo.ar",
                            "data/MT/ldc_web_eg.test.lev.0.jo.en",
                        ),
                        (
                            "data/MT/ldc_web_eg.test.lev.0.ps.ar",
                            "data/MT/ldc_web_eg.test.lev.0.ps.en",
                        ),
                        (
                            "data/MT/ldc_web_eg.test.lev.0.sy.ar",
                            "data/MT/ldc_web_eg.test.lev.0.sy.en",
                        ),
                    ],
                },
                "MDC-MGR": {
                    "test": (
                        "data/MT/ldc_web_eg.test.mgr.0.tn.ar",
                        "data/MT/ldc_web_eg.test.mgr.0.tn.en",
                    )
                },
                "MDC-MSA": {
                    "test": (
                        "data/MT/ldc_web_eg.test.msa.0.ms.ar",
                        "data/MT/ldc_web_eg.test.msa.0.ms.en",
                    )
                },
                "Media-Gulf": {
                    "test": (
                        "data/MT/summa-Oman.test.glf.0.om.ar",
                        "data/MT/summa-Oman.test.glf.0.om.en",
                    )
                },
                "Media-LEV": {
                    "test": (
                        "data/MT/summa-LBC.test.lev.0.lb.ar",
                        "data/MT/summa-LBC.test.lev.0.lb.en",
                    )
                },
                "Media-MGR": {
                    "test": (
                        "data/MT/summa-2M.test.mgr.0.ma.ar",
                        "data/MT/summa-2M.test.mgr.0.ma.en",
                    )
                },
                "Media-MSA": {
                    "test": [
                        (
                            "data/MT/summa-AJ.test.msa.0.ms.ar",
                            "data/MT/summa-AJ.test.msa.0.ms.en",
                        ),
                        (
                            "data/MT/summa-BBC.test.msa.0.ms.ar",
                            "data/MT/summa-BBC.test.msa.0.ms.en",
                        ),
                    ]
                },
                "QAraC-Gulf": {
                    "dev": (
                        "data/MT/QAraC.dev.glf.0.qa.ar",
                        "data/MT/QAraC.dev.glf.0.qa.ar",
                    ),
                    "test": (
                        "data/MT/QAraC.test.glf.0.qa.ar",
                        "data/MT/QAraC.test.glf.0.qa.en",
                    ),
                },
                "default": [
                    "APT-LEV",
                    "APT-Nile",
                    "MADAR-Gulf",
                    "MADAR-LEV",
                    "MADAR-MGR",
                    "MADAR-MSA",
                    "MADAR-Nile",
                    "MDC-LEV",
                    "MDC-MGR",
                    "MDC-MSA",
                    "Bible-MGR",
                    "Bible-MSA",
                ],
            },
            "task_type": TaskType.SequenceToSequence,
        }

    def get_data_sample(self):
        return {"input": "Sentence in language #1", "label": "Sentence in language #2"}

    def load_data(self, data_path, no_labels=False):
        data = []

        with open(data_path + self.src, "r") as fpsrc, open(
            data_path + self.tgt, "r"
        ) as fptgt:
            for line_idx, (srcline, tgtline) in enumerate(zip(fpsrc, fptgt)):
                data.append(
                    {
                        "input": srcline.strip(),
                        "label": tgtline.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
