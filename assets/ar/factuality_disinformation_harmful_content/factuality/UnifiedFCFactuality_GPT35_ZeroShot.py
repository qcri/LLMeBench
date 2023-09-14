from llmebench.datasets import UnifiedFCFactualityDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import FactualityTask


def config():
    return {
        "dataset": UnifiedFCFactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "class_labels": ["true", "false"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/factuality_stance_ramy/ramy_arabic_fact_checking.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"Classify the text as only true or false. Provide only label.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
    )
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": prompt_string,
            }
        ],
    }


def post_process(response):
    label = response["choices"][0]["text"].lower().replace(".", "").lower()

    if (
        label.startswith("I am unable to verify".lower())
        or label.startswith("I am unable to categorize".lower())
        or label.startswith(
            "I am an AI language model and I am unable to verify".lower()
        )
    ):
        label_fixed = None
    elif "label: incorrect" in label or "incorrect" in label:
        label_fixed = "false"
    elif "label: correct" in label or "correct" in label:
        label_fixed = "true"
    elif "true" == label or "false" == label:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
