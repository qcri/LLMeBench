from llmebench.datasets import COVID19FactualityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import FactualityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Weighted-F1": "0.497"},
    }


def config():
    return {
        "dataset": COVID19FactualityDataset,
        "dataset_args": {},
        "task": FactualityTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["yes", "no"],
            "max_tries": 30,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
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
    base_prompt = f'Annotate the "tweet" into one of the following categories: yes or no. Provide only label.'
    return [
        {
            "role": "system",
            "content": "You are a social media expert, a fact-checker and you can annotate tweets.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    if (
        "label: incorrect" in label
        or "incorrect" in label
        or label == "no"
        or "label: no" in label
    ):
        label_fixed = "no"
    elif (
        "label: correct" in label
        or "correct" in label
        or label == "yes"
        or "label: yes" in label
    ):
        label_fixed = "yes"
    else:
        label_fixed = None

    return label_fixed
