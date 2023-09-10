from llmebench.datasets import AdultDataset
from llmebench.models import PetalsModel
from llmebench.tasks import AdultTask


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 10,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/adult/adult-test.tsv",
        },
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert annotator, you can identify and label adult content within a tweet.\n"
        f"Given the following tweet, label it as 'ADULT' or 'NOT_ADULT' based on the content of the tweet\n"
        f"Provide only label.\n\n"
        f"tweet: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    label_list = config()["model_args"]["class_labels"]

    if label in label_list:
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
