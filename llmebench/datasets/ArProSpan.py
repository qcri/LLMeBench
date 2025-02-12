import json
from pathlib import Path

from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ArProSpanDataset(DatasetBase):
    def __init__(self, techniques_path=None, **kwargs):
        self.techniques_path = Path(techniques_path) if techniques_path else None
        super(ArProSpanDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """
                to add
            """,
            "link": "",
            "license": "",
            "splits": {
                "test": "ArMPro_span_test.jsonl",
                "dev": "ArMPro_span_dev.jsonl",
                "train": "ArMPro_span_train.jsonl",
            },
            "task_type": TaskType.SequenceLabeling,
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
        return {
            "input_id": "001",
            "input": "paragraph",
            "label": [{"technique": "Guilt_by_Association", "start": 13, "end": 52}],
            "line_number": 1,
        }

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
        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                line_data = json.loads(line)
                id = line_data.get("paragraph_id", "")
                text = line_data.get("paragraph", "")
                label = line_data.get("labels", [])

                # we need to par text at evaluation to do some matching against predicted spans
                for l in label:
                    l["par_txt"] = text

                data.append(
                    {
                        "input": text,
                        "input_id": id,
                        "label": label,
                        "line_number": line_idx,
                    }
                )

        print("loaded %d docs from file..." % len(data))

        return data
