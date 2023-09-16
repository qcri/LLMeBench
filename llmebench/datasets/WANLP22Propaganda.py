import json

from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class WANLP22PropagandaDataset(DatasetBase):
    def __init__(self, techniques_path=None, **kwargs):
        # Get the path to the file listing the target techniques
        self.techniques_path = Path(techniques_path) if techniques_path else None
        super(WANLP22PropagandaDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@inproceedings{alam2022overview,
              title={Overview of the $\\{$WANLP$\\}$ 2022 Shared Task on Propaganda Detection in $\\{$A$\\}$rabic},
              author={Alam, Firoj and Mubarak, Hamdy and Zaghouani, Wajdi and Da San Martino, Giovanni and Nakov, Preslav and others},
              booktitle={Proceedings of the The Seventh Arabic Natural Language Processing Workshop (WANLP)},
              pages={108--118},
              year={2022},
              organization={Association for Computational Linguistics}
            }""",
            "link": "https://gitlab.com/araieval/propaganda-detection",
            "license": "Research Purpose Only",
            "splits": {
                "test": "data/factuality_disinformation_harmful_content/propaganda/task1_test_gold_label_final.json",
                "train": "data/factuality_disinformation_harmful_content/propaganda/task1_train.json",
            },
            "task_type": TaskType.MultiLabelClassification,
            "class_labels": [
                "no technique",
                "Smears",
                "Exaggeration/Minimisation",
                "Loaded Language",
                "Appeal to fear/prejudice",
                "Name calling/Labeling",
                "Slogans",
                "Repetition",
                "Doubt",
                "Obfuscation, Intentional vagueness, Confusion",
                "Flag-waving",
                "Glittering generalities (Virtue)",
                "Misrepresentation of Someone's Position (Straw Man)",
                "Presenting Irrelevant Data (Red Herring)",
                "Appeal to authority",
                "Whataboutism",
                "Black-and-white Fallacy/Dictatorship",
                "Thought-terminating cliché",
                "Causal Oversimplification",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {"input": "Tweet", "label": ["no technique"]}

    def get_predefined_techniques(self):
        # Load a pre-defined list of propaganda techniques, if available
        if self.techniques_path and self.techniques_path.exists():
            with open(self.techniques_path, "r", encoding="utf-8") as f:
                techniques = [label.strip() for label in f.readlines()]
        else:
            techniques = [
                "no technique",
                "Smears",
                "Exaggeration/Minimisation",
                "Loaded Language",
                "Appeal to fear/prejudice",
                "Name calling/Labeling",
                "Slogans",
                "Repetition",
                "Doubt",
                "Obfuscation, Intentional vagueness, Confusion",
                "Flag-waving",
                "Glittering generalities (Virtue)",
                "Misrepresentation of Someone's Position (Straw Man)",
                "Presenting Irrelevant Data (Red Herring)",
                "Appeal to authority",
                "Whataboutism",
                "Black-and-white Fallacy/Dictatorship",
                "Thought-terminating cliché",
                "Causal Oversimplification",
            ]

        return techniques

    def load_data(self, data_path):
        data = []
        with open(data_path, mode="r", encoding="utf-8") as infile:
            json_data = json.load(infile)
            for index, tweet in enumerate(json_data):
                text = tweet["text"]
                label = tweet["labels"]
                data.append({"input": text, "label": label, "line_number": index})

        return data
