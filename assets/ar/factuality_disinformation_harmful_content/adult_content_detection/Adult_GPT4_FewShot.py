from llmebench.datasets import AdultDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AdultTask


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 30,
        },
        "general_args": {
            "fewshot": {
                "deduplicate": True,
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + example["label"]
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = f'Given the following tweet, label it as "ADULT" or "NOT_ADULT" based on the content of the tweet'
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label adult content within a tweet.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    if not response:
        return None

    label = response["choices"][0]["message"]["content"]

    if "label: NOT_ADULT" in label or "NOT_ADULT" == label:
        label_fixed = "NOT_ADULT"
    elif "label: ADULT" in label or "ADULT" == label:
        label_fixed = "ADULT"
    else:
        label_fixed = None

    return label_fixed
