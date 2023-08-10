import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class PropagandaSemEval23Dataset(DatasetBase):
    def __init__(self, techniques_path=None, **kwargs):
        # Get the path to the file listing the target techniques
        self.techniques_path = Path(techniques_path) if techniques_path else None
        super(PropagandaSemEval23Dataset, self).__init__(**kwargs)

    def citation(self):
        return """@article{wanlp2023,
          year={2023}
        }"""

    def get_data_sample(self):
        return {"input": {"text": "text"}, "label": ["no_technique"]}

    def get_predefined_techniques(self):
        # Load a pre-defined list of propaganda techniques, if available
        if self.techniques_path and self.techniques_path.exists():
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
