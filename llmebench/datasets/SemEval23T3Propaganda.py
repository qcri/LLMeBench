import json
from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class SemEval23T3PropagandaDataset(DatasetBase):
    def __init__(self, techniques_path=None, **kwargs):
        # Get the path to the file listing the target techniques
        self.techniques_path = Path(techniques_path) if techniques_path else None
        super(SemEval23T3PropagandaDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": ["en", "es", "fr", "de", "el", "it", "ka", "pl", "ru"],
            "citation": """@inproceedings{piskorski-etal-2023-semeval,
                title = "{S}em{E}val-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup",
                author = "Piskorski, Jakub  and
                  Stefanovitch, Nicolas  and
                  Da San Martino, Giovanni  and
                  Nakov, Preslav",
                booktitle = "Proceedings of the 17th International Workshop on Semantic Evaluation (SemEval-2023)",
                month = jul,
                year = "2023",
                address = "Toronto, Canada",
                publisher = "Association for Computational Linguistics",
                url = "https://aclanthology.org/2023.semeval-1.317",
                doi = "10.18653/v1/2023.semeval-1.317",
                pages = "2343--2361",
                abstract = "We describe SemEval-2023 task 3 on Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multilingual Setup: the dataset, the task organization process, the evaluation setup, the results, and the participating systems. The task focused on news articles in nine languages (six known to the participants upfront: English, French, German, Italian, Polish, and Russian), and three additional ones revealed to the participants at the testing phase: Spanish, Greek, and Georgian). The task featured three subtasks: (1) determining the genre of the article (opinion, reporting, or satire), (2) identifying one or more frames used in an article from a pool of 14 generic frames, and (3) identify the persuasion techniques used in each paragraph of the article, using a taxonomy of 23 persuasion techniques. This was a very popular task: a total of 181 teams registered to participate, and 41 eventually made an official submission on the test set.",
            }""",
            "link": "https://propaganda.math.unipd.it/semeval2023task3/",
            "splits": {
                "de": {
                    "dev": "ge_dev_subtask3.json",
                    "train": "ge_train_subtask3.json",
                },
                "en": {
                    "dev": "en_dev_subtask3.json",
                    "train": "en_train_subtask3.json",
                },
                "fr": {
                    "dev": "fr_dev_subtask3.json",
                    "train": "fr_train_subtask3.json",
                },
                "it": {
                    "dev": "it_dev_subtask3.json",
                    "train": "it_train_subtask3.json",
                },
                "pl": {
                    "dev": "po_dev_subtask3.json",
                    "train": "po_train_subtask3.json",
                },
                "ru": {
                    "dev": "ru_dev_subtask3.json",
                    "train": "ru_train_subtask3.json",
                },
            },
            "task_type": TaskType.MultiLabelClassification,
            "class_labels": [
                "Appeal_to_Authority",
                "Appeal_to_Fear-Prejudice",
                "Appeal_to_Hypocrisy",
                "Appeal_to_Popularity",
                "Appeal_to_Time",
                "Appeal_to_Values",
                "Causal_Oversimplification",
                "Consequential_Oversimplification",
                "Conversation_Killer",
                "Doubt",
                "Exaggeration-Minimisation",
                "False_Dilemma-No_Choice",
                "Flag_Waving",
                "Guilt_by_Association",
                "Loaded_Language",
                "Name_Calling-Labeling",
                "Obfuscation-Vagueness-Confusion",
                "Questioning_the_Reputation",
                "Red_Herring",
                "Repetition",
                "Slogans",
                "Straw_Man",
                "Whataboutism",
                "no_technique",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "text", "label": ["no_technique"], "line_number": 0}

    def get_predefined_techniques(self):
        # Load a pre-defined list of propaganda techniques, if available
        if self.techniques_path and self.techniques_path.exists():
            self.techniques_path = self.resolve_path(self.techniques_path)
            with open(self.techniques_path, "r", encoding="utf-8") as f:
                techniques = [label.strip() for label in f.readlines()]
        else:
            techniques = [
                "Appeal_to_Authority",
                "Appeal_to_Fear-Prejudice",
                "Appeal_to_Hypocrisy",
                "Appeal_to_Popularity",
                "Appeal_to_Time",
                "Appeal_to_Values",
                "Causal_Oversimplification",
                "Consequential_Oversimplification",
                "Conversation_Killer",
                "Doubt",
                "Exaggeration-Minimisation",
                "False_Dilemma-No_Choice",
                "Flag_Waving",
                "Guilt_by_Association",
                "Loaded_Language",
                "Name_Calling-Labeling",
                "Obfuscation-Vagueness-Confusion",
                "Questioning_the_Reputation",
                "Red_Herring",
                "Repetition",
                "Slogans",
                "Straw_Man",
                "Whataboutism",
                "no_technique",
            ]

        return techniques

    def load_data(self, data_path):
        data_path = self.resolve_path(data_path)

        data = []
        with open(data_path, mode="r", encoding="utf-8") as infile:
            json_data = json.load(infile)
            for index, tweet in enumerate(json_data):
                text = tweet["text"]
                label = tweet["label"]
                if "no-technique" in label:
                    index = label.index("no-technique")
                    label[index] = "no_technique"
                data.append({"input": text, "label": label, "line_number": index})

        return data
