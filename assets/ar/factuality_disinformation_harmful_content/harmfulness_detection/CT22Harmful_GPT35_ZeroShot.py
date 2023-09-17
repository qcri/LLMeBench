from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import HarmfulDetectionTask


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "dataset_args": {},
        "task": HarmfulDetectionTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": "Classify the sentence as harmful or not harmful. Answer only by yes or no. Provide only "
                "label.\n\n" + "sentence: " + input_sample + "label: \n",
            }
        ],
    }


def post_process(response):
    pred_label = response["choices"][0]["text"]
    pred_label = pred_label.replace(".", "").strip().lower()

    if pred_label.startswith("harmful") or pred_label.startswith("yes"):
        pred_label = "1"

    if (
        pred_label.startswith("no")
        or pred_label == "label: safe"
        or "not harmful" in pred_label
    ):
        pred_label = "0"

    return pred_label
