from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import OpenAIModel
from llmebench.tasks import OffensiveTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.874"},
    }


def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "task": OffensiveTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Given a tweet, predict whether it is offensive or not. Answer only by using"
        " offensive and not_offensive. Here are some examples:\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label offensive content within a tweet.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        # label = "no" if example["label"] == "0" else "yes"
        label = "not_offensive" if example["label"] == "NOT_OFF" else "offensive"

        out_prompt = (
            out_prompt + "Tweet: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Tweet: " + input_sample + "\nLabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]

    if "not_offensive" in out:
        out = "NOT_OFF"
    elif "offensive" in out:
        out = "OFF"
    else:
        out = None
    return out
