import json

from llmebench.datasets.SQuADBase import SQuADBase
from llmebench.tasks import TaskType


class ARCDDataset(SQuADBase):
    def __init__(self, **kwargs):
        super(ARCDDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{mozannar-etal-2019-neural,
                title = "Neural {A}rabic Question Answering",
                author = "Mozannar, Hussein  and
                  Maamary, Elie  and
                  El Hajal, Karl  and
                  Hajj, Hazem",
                booktitle = "Proceedings of the Fourth Arabic Natural Language Processing Workshop",
                month = aug,
                year = "2019",
                address = "Florence, Italy",
                publisher = "Association for Computational Linguistics",
                url = "https://www.aclweb.org/anthology/W19-4612",
                doi = "10.18653/v1/W19-4612",
                pages = "108--118",
                abstract = "This paper tackles the problem of open domain factual Arabic question answering (QA) using Wikipedia as our knowledge source. This constrains the answer of any question to be a span of text in Wikipedia. Open domain QA for Arabic entails three challenges: annotated QA datasets in Arabic, large scale efficient information retrieval and machine reading comprehension. To deal with the lack of Arabic QA datasets we present the Arabic Reading Comprehension Dataset (ARCD) composed of 1,395 questions posed by crowdworkers on Wikipedia articles, and a machine translation of the Stanford Question Answering Dataset (Arabic-SQuAD). Our system for open domain question answering in Arabic (SOQAL) is based on two components: (1) a document retriever using a hierarchical TF-IDF approach and (2) a neural reading comprehension model using the pre-trained bi-directional transformer BERT. Our experiments on ARCD indicate the effectiveness of our approach with our BERT-based reader achieving a 61.3 F1 score, and our open domain system SOQAL achieving a 27.6 F1 score.",
            }""",
            "link": "https://github.com/husseinmozannar/SOQAL",
            "license": "MIT License",
            "splits": {
                "test": "arcd-test.json",
                "train": "arcd-train.json",
            },
            "task_type": TaskType.QuestionAnswering,
        }
