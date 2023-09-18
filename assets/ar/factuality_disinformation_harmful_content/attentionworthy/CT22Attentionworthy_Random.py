from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import RandomModel
from llmebench.tasks import AttentionworthyTask, TaskType


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Random",
        "description": "Random Baseline.",
        "scores": {"Weighted-F1": "0.125"},
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "dataset_args": {},
        "task": AttentionworthyTask,
        "task_args": {},
        "model": RandomModel,
        "model_args": {
            "task_type": TaskType.Classification,
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    return input_sample


def post_process(response):
    return response["random_response"]
