import json

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArAIEVAL232B(DatasetBase):
    def __init__(self, **kwargs):
        super(ArAIEVAL232B, self).__init__(**kwargs)

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
                "test": "ArAiEval23_disinfo_subtask2B_test.jsonl",
                "dev": "ArAiEval23_disinfo_subtask2B_dev.jsonl",
                "train": "ArAiEval23_disinfo_subtask2B_train.jsonl",
            },
            "task_type": TaskType.Classification,
            "class_labels": ["HS", "OFF", "SPAM", "Rumor"],
        }

    @staticmethod
    def get_data_sample():
        return {"id": "001", "input": "text", "label": "HS", "type": "tweet"}

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
