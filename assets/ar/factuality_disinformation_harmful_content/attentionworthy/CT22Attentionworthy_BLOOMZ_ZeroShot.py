from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import PetalsModel
from llmebench.tasks import AttentionworthyTask


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "dataset_args": {},
        "task": AttentionworthyTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt = (
        f"Predict whether a tweet should get the attention of policy makers. Use the following categories: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action\n\n"
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return {
        "prompt": prompt,
    }


def post_process(response):
    label = response["outputs"].lower().replace("<s>", "").replace("</s>", "").strip()
    label_fixed = None

    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif label == "yes_discusses_covid-19_vaccine_side_effects":
        label_fixed = "yes_discusses_cure"
    elif label == "yes_harmful":
        label_fixed = "harmful"
    elif label.startswith("yes"):
        label_fixed = label

    return label_fixed
