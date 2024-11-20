from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.745"},
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar/dev", "fewshot": {"train_split": "ar"}},
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Classify the 'sentence' as subjective or objective. Provide only label.\n"
    )
    return [
        {
            "role": "system",
            "content": "You are social media expert. You can annotate important tweets and require attention from journalists, fact-checker, and government entities.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        label = "objective" if example["label"] == "OBJ" else "subjective"

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "sentence: "
            + example["input"]
            + "\nlabel: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "sentence: " + input_sample + "\nlabel: \n"

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "label: objective" in label:
        label_fixed = "OBJ"
    elif "label: subjective" in label:
        label_fixed = "SUBJ"
    elif label == "objective" or label == "objective.":
        label_fixed = "OBJ"

    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
