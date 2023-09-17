from llmebench.datasets import Q2QSimDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import Q2QSimDetectionTask


def config():
    return {
        "dataset": Q2QSimDataset,
        "dataset_args": {},
        "task": Q2QSimDetectionTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    q1, q2 = input_sample.split("\t")
    prompt = f"Are the two questions below semantically similar? The output should be exactly in form yes or no.\n\nQ1: {q1}\nQ2: {q2}\nlabel: "
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt,
            }
        ],
    }


def post_process(response):
    input_label = response["choices"][0]["text"]
    input_label = input_label.replace(".", "").strip().lower()
    pred_label = ""

    if "yes" in input_label or "label: 1" in input_label:
        pred_label = "1"
    if (
        input_label == "no"
        or input_label.startswith("no,")
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not semantically similar" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
