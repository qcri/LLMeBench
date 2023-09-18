from llmebench.datasets import SemEval23T3PropagandaDataset
from llmebench.models import RandomModel
from llmebench.tasks import MultilabelPropagandaTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Micro-F1": "0.123"},
    }


def config():
    return {
        "dataset": SemEval23T3PropagandaDataset,
        "dataset_args": {"techniques_path": "techniques_subtask3.txt"},
        "task": MultilabelPropagandaTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
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
        },
        "general_args": {"test_split": "de/dev"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    random_labels = response["random_response"]
    if "no_technique" in random_labels:
        return ["no_technique"]
    else:
        return random_labels
