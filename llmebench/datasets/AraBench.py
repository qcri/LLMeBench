from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class AraBenchDataset(DatasetBase):
    def __init__(self, src_lang, tgt_lang, **kwargs):
        super(AraBenchDataset, self).__init__(**kwargs)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    @staticmethod
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
                "APT-LEV_ldc_web_lv.lev.0.lv": {
                    "dev": "data/MT/ldc_web_lv.dev.lev.0.lv",
                    "test": "data/MT/ldc_web_lv.test.lev.0.lv",
                    "train": "data/MT/ldc_web_lv.train.lev.0.lv",
                },
                "APT-Nile_ldc_web_eg.nil.0.eg": {
                    "dev": "data/MT/ldc_web_eg.dev.nil.0.eg",
                    "test": "data/MT/ldc_web_eg.test.nil.0.eg",
                    "train": "data/MT/ldc_web_eg.train.nil.0.eg",
                },
                "Bible-MGR_bible.mgr.0.ma": {
                    "dev": "data/MT/bible.dev.mgr.0.ma",
                    "test": "data/MT/bible.test.mgr.0.ma",
                    "train": "data/MT/bible.train.mgr.0.ma",
                },
                "Bible-MGR_bible.mgr.0.tn": {
                    "dev": "data/MT/bible.dev.mgr.0.tn",
                    "test": "data/MT/bible.test.mgr.0.tn",
                    "train": "data/MT/bible.train.mgr.0.tn",
                },
                "Bible-MSA_bible.msa.0.ms": {
                    "dev": "data/MT/bible.dev.msa.0.ms",
                    "test": "data/MT/bible.test.msa.0.ms",
                    "train": "data/MT/bible.train.msa.0.ms",
                },
                "Bible-MSA_bible.msa.1.ms": {
                    "dev": "data/MT/bible.dev.msa.1.ms",
                    "test": "data/MT/bible.test.msa.1.ms",
                    "train": "data/MT/bible.train.msa.1.ms",
                },
                "MADAR-Gulf_madar.glf.0.iq": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.0.iq",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.1.iq": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.1.iq",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.2.iq": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.2.iq",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.0.om": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.0.om",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.0.qa": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.0.qa",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.0.sa": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.0.sa",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.1.sa": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.1.sa",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-Gulf_madar.glf.0.ye": {
                    "dev": "data/MT/madar.dev.glf.0.qa",
                    "test": "data/MT/madar.test.glf.0.ye",
                    "train": "data/MT/madar.train.glf.0.qa",
                },
                "MADAR-LEV_madar.lev.0.jo": {
                    "dev": "data/MT/madar.dev.lev.0.lb",
                    "test": "data/MT/madar.test.lev.0.jo",
                    "train": "data/MT/madar.train.lev.0.lb",
                },
                "MADAR-LEV_madar.lev.1.jo": {
                    "dev": "data/MT/madar.dev.lev.0.lb",
                    "test": "data/MT/madar.test.lev.1.jo",
                    "train": "data/MT/madar.train.lev.0.lb",
                },
                "MADAR-LEV_madar.lev.0.lb": {
                    "dev": "data/MT/madar.dev.lev.0.lb",
                    "test": "data/MT/madar.test.lev.0.lb",
                    "train": "data/MT/madar.train.lev.0.lb",
                },
                "MADAR-LEV_madar.lev.0.pa": {
                    "dev": "data/MT/madar.dev.lev.0.lb",
                    "test": "data/MT/madar.test.lev.0.pa",
                    "train": "data/MT/madar.train.lev.0.lb",
                },
                "MADAR-LEV_madar.lev.0.sy": {
                    "dev": "data/MT/madar.dev.lev.0.lb",
                    "test": "data/MT/madar.test.lev.0.sy",
                    "train": "data/MT/madar.train.lev.0.lb",
                },
                "MADAR-LEV_madar.lev.1.sy": {
                    "dev": "data/MT/madar.dev.lev.0.lb",
                    "test": "data/MT/madar.test.lev.1.sy",
                    "train": "data/MT/madar.train.lev.0.lb",
                },
                "MADAR-MGR_madar.mgr.0.dz": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.0.dz",
                    "train": "data/MT/madar.train.mgr.0.ma",
                },
                "MADAR-MGR_madar.mgr.0.ly": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.0.ly",
                    "train": "data/MT/madar.train.mgr.0.ma",
                },
                "MADAR-MGR_madar.mgr.1.ly": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.1.ly",
                    "train": "data/MT/madar.train.mgr.0.ma",
                },
                "MADAR-MGR_madar.mgr.0.ma": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.0.ma",
                    "train": "data/MT/madar.train.mgr.0.ma",
                },
                "MADAR-MGR_madar.mgr.1.ma": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.1.ma",
                    "train": "data/MT/madar.train.mgr.0.ma",
                },
                "MADAR-MGR_madar.mgr.0.tn": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.0.tn",
                    "train": "data/MT/madar.train.mgr.0.tn",
                },
                "MADAR-MGR_madar.mgr.1.tn": {
                    "dev": "data/MT/madar.dev.mgr.0.ma",
                    "test": "data/MT/madar.test.mgr.1.tn",
                    "train": "data/MT/madar.train.mgr.0.tn",
                },
                "MADAR-MSA_madar.msa.0.ms": {
                    "dev": "data/MT/madar.dev.msa.0.ms",
                    "test": "data/MT/madar.test.msa.0.ms",
                    "train": "data/MT/madar.train.msa.0.ms",
                },
                "MADAR-Nile_madar.nil.0.eg": {
                    "dev": "data/MT/madar.dev.nil.0.eg",
                    "test": "data/MT/madar.test.nil.0.eg",
                    "train": "data/MT/madar.train.nil.0.eg",
                },
                "MADAR-Nile_madar.nil.1.eg": {
                    "dev": "data/MT/madar.dev.nil.0.eg",
                    "test": "data/MT/madar.test.nil.1.eg",
                    "train": "data/MT/madar.train.nil.0.eg",
                },
                "MADAR-Nile_madar.nil.2.eg": {
                    "dev": "data/MT/madar.dev.nil.0.eg",
                    "test": "data/MT/madar.test.nil.2.eg",
                    "train": "data/MT/madar.train.nil.0.eg",
                },
                "MADAR-Nile_madar.nil.0.sd": {
                    "dev": "data/MT/madar.dev.nil.0.eg",
                    "test": "data/MT/madar.test.nil.0.sd",
                    "train": "data/MT/madar.train.nil.0.eg",
                },
                "MDC-LEV_ldc_web_eg.lev.0.jo": {
                    "dev": "data/MT/ldc_web_eg.dev.lev.0.sy",
                    "test": "data/MT/ldc_web_eg.test.lev.0.jo",
                },
                "MDC-LEV_ldc_web_eg.lev.0.ps": {
                    "dev": "data/MT/ldc_web_eg.dev.lev.0.sy",
                    "test": "data/MT/ldc_web_eg.test.lev.0.ps",
                },
                "MDC-LEV_ldc_web_eg.lev.0.sy": {
                    "dev": "data/MT/ldc_web_eg.dev.lev.0.sy",
                    "test": "data/MT/ldc_web_eg.test.lev.0.sy",
                },
                "MDC-MGR_ldc_web_eg.mgr.0.tn": {
                    "test": "data/MT/ldc_web_eg.test.mgr.0.tn",
                },
                "MDC-MSA_ldc_web_eg.msa.0.ms": {
                    "test": "data/MT/ldc_web_eg.test.msa.0.ms",
                },
                "Media-Gulf_summa-Oman.glf.0.om": {
                    "test": "data/MT/summa-Oman.test.glf.0.om",
                },
                "Media-LEV_summa-LBC.lev.0.lb": {
                    "test": "data/MT/summa-LBC.test.lev.0.lb",
                },
                "Media-MGR_summa-2M.mgr.0.ma": {
                    "test": "data/MT/summa-2M.test.mgr.0.ma",
                },
                "Media-MSA_summa-AJ.msa.0.ms": {
                    "test": "data/MT/summa-AJ.test.msa.0.ms",
                },
                "Media-MSA_summa-BBC.msa.0.ms": {
                    "test": "data/MT/summa-BBC.test.msa.0.ms",
                },
                "QAraC-Gulf_QAraC.glf.0.qa": {
                    "dev": "data/MT/QAraC.dev.glf.0.qa",
                    "test": "data/MT/QAraC.test.glf.0.qa",
                },
                "default": [
                    "APT-LEV_ldc_web_lv.lev.0.lv",
                    "APT-Nile_ldc_web_eg.nil.0.eg",
                    "Bible-MGR_bible.mgr.0.ma",
                    "Bible-MGR_bible.mgr.0.tn",
                    "Bible-MSA_bible.msa.0.ms",
                    "Bible-MSA_bible.msa.1.ms",
                    "MADAR-Gulf_madar.glf.0.iq",
                    "MADAR-Gulf_madar.glf.1.iq",
                    "MADAR-Gulf_madar.glf.2.iq",
                    "MADAR-Gulf_madar.glf.0.om",
                    "MADAR-Gulf_madar.glf.0.qa",
                    "MADAR-Gulf_madar.glf.0.sa",
                    "MADAR-Gulf_madar.glf.1.sa",
                    "MADAR-Gulf_madar.glf.0.ye",
                    "MADAR-LEV_madar.lev.0.jo",
                    "MADAR-LEV_madar.lev.1.jo",
                    "MADAR-LEV_madar.lev.0.lb",
                    "MADAR-LEV_madar.lev.0.pa",
                    "MADAR-LEV_madar.lev.0.sy",
                    "MADAR-LEV_madar.lev.1.sy",
                    "MADAR-MGR_madar.mgr.0.dz",
                    "MADAR-MGR_madar.mgr.0.ly",
                    "MADAR-MGR_madar.mgr.1.ly",
                    "MADAR-MGR_madar.mgr.0.ma",
                    "MADAR-MGR_madar.mgr.1.ma",
                    "MADAR-MGR_madar.mgr.0.tn",
                    "MADAR-MGR_madar.mgr.1.tn",
                    "MADAR-MSA_madar.msa.0.ms",
                    "MADAR-Nile_madar.nil.0.eg",
                    "MADAR-Nile_madar.nil.1.eg",
                    "MADAR-Nile_madar.nil.2.eg",
                    "MADAR-Nile_madar.nil.0.sd",
                    "MDC-LEV_ldc_web_eg.lev.0.jo",
                    "MDC-LEV_ldc_web_eg.lev.0.ps",
                    "MDC-LEV_ldc_web_eg.lev.0.sy",
                    "MDC-MGR_ldc_web_eg.mgr.0.tn",
                    "MDC-MSA_ldc_web_eg.msa.0.ms",
                ],
            },
            "task_type": TaskType.SequenceToSequence,
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Sentence in language #1", "label": "Sentence in language #2"}

    def load_data(self, data_path, no_labels=False):
        data = []

        with open(f"{data_path}.{self.src_lang}", "r") as fpsrc, open(
            f"{data_path}.{self.tgt_lang}", "r"
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
