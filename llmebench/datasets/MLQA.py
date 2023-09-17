from llmebench.datasets.SQuADBase import SQuADBase
from llmebench.tasks import TaskType


class MLQADataset(SQuADBase):
    def __init__(self, **kwargs):
        super(MLQADataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{lewis-etal-2020-mlqa,
                title = "{MLQA}: Evaluating Cross-lingual Extractive Question Answering",
                author = "Lewis, Patrick  and
                  Oguz, Barlas  and
                  Rinott, Ruty  and
                  Riedel, Sebastian  and
                  Schwenk, Holger",
                booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
                month = jul,
                year = "2020",
                address = "Online",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2020.acl-main.653",
                doi = "10.18653/v1/2020.acl-main.653",
                pages = "7315--7330",
                abstract = "Question answering (QA) models have shown rapid progress enabled by the availability of large, high-quality benchmark datasets. Such annotated datasets are difficult and costly to collect, and rarely exist in languages other than English, making building QA systems that work well in other languages challenging. In order to develop such systems, it is crucial to invest in high quality multilingual evaluation benchmarks to measure progress. We present MLQA, a multi-way aligned extractive QA evaluation benchmark intended to spur research in this area. MLQA contains QA instances in 7 languages, English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA has over 12K instances in English and 5K in each other language, with each instance parallel between 4 languages on average. We evaluate state-of-the-art cross-lingual models and machine-translation-based baselines on MLQA. In all cases, transfer results are shown to be significantly behind training-language performance.",
            }""",
            "link": "https://github.com/facebookresearch/mlqa",
            "license": "CC BY-NC 4.0",
            "splits": {
                "dev": "dev/dev-context-ar-question-ar.json",
                "test": "test/test-context-ar-question-ar.json",
            },
            "task_type": TaskType.QuestionAnswering,
        }
