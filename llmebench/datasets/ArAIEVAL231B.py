import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArAIEVAL231B(DatasetBase):
    def __init__(self, **kwargs):
        super(ArAIEVAL231B, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
                @inproceedings{araieval:arabicnlp2023-overview,
                    title = "ArAIEval Shared Task: Persuasion Techniques and Disinformation Detection in Arabic Text",
                    author = "Hasanain, Maram and Alam, Firoj and Mubarak, Hamdy, and Abdaljalil, Samir  and Zaghouani, Wajdi and Nakov, Preslav  and Da San Martino, Giovanni and Freihat, Abed Alhakim",
                    booktitle = "Proceedings of the First Arabic Natural Language Processing Conference (ArabicNLP 2023)",
                    month = Dec,
                    year = "2023",
                    address = "Singapore",
                    publisher = "Association for Computational Linguistics",
                }
                
                @inproceedings{alam-etal-2022-overview,
                    title = "Overview of the {WANLP} 2022 Shared Task on Propaganda Detection in {A}rabic",
                    author = "Alam, Firoj  and
                      Mubarak, Hamdy  and
                      Zaghouani, Wajdi  and
                      Da San Martino, Giovanni  and
                      Nakov, Preslav",
                    booktitle = "Proceedings of the Seventh Arabic Natural Language Processing Workshop (WANLP)",
                    month = dec,
                    year = "2022",
                    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
                    publisher = "Association for Computational Linguistics",
                    url = "https://aclanthology.org/2022.wanlp-1.11",
                    pages = "108--118",
                }
                
                @article{10.3389/frai.2023.1219767,
                  author    = {Hamdy Mubarak and Samir Abdaljalil and Azza Nassar and Firoj Alam},
                  title     = {Detecting and identifying the reasons for deleted tweets before they are posted},
                  journal   = {Frontiers in Artificial Intelligence},
                  volume    = {6},
                  year      = {2023},
                  url       = {https://www.frontiersin.org/articles/10.3389/frai.2023.1219767},
                  doi       = {10.3389/frai.2023.1219767},
                  issn      = {2624-8212},  
                }

            """,
            "link": "https://gitlab.com/araieval/wanlp2023_araieval/",
            "license": "CC BY-NC-SA 2.0",
            "splits": {
                "test": "task1B_test.jsonl",
                "dev": "task1B_dev.jsonl",
                "train": "task1B_train.jsonl",
            },
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "Appeal to Authority",
                "Appeal to Fear/Prejudice",
                "Appeal to Hypocrisy",
                "Appeal to Time",
                "Appeal to values",
                "Bandwagon",
                "Black-and-white Fallacy/Dictatorship",
                "Casting Doubt",
                "Causal Oversimplification",
                "Consequential Oversimplification",
                "Exaggeration/Minimisation",
                "Flag Waving",
                "Loaded Language",
                "Misrepresentation of Someone's Position (Strawman)",
                "Name Calling",
                "Obfuscation, Intentional vagueness, Confusion",
                "Presenting Irrelevant Data (Red Herring)",
                "Reductio ad Hitlerum",
                "Repetition",
                "Slogans",
                "Smears",
                "Thought-terminating cliché",
                "Whataboutism",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"id": "001", "input": "tweet", "label": "true", "type": "paragraph"}

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                line_data = json.loads(line)
                id = line_data.get("id", None)
                text = line_data.get("text", "")
                label = line_data.get("label", "").lower()
                data.append({"input": text, "label": label, "line_number": id})

        return data
