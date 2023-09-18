from llmebench.datasets import ASNDDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import NewsCategorizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.594"},
    }


def config():
    return {
        "dataset": ASNDDataset,
        "task": NewsCategorizationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "crime-war-conflict",
                "spiritual",
                "health",
                "politics",
                "human-rights-press-freedom",
                "education",
                "business-and-economy",
                "art-and-entertainment",
                "others",
                "science-and-technology",
                "sports",
                "environment",
            ],
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
            + "article: "
            + example["input"]
            + "\ncategory: "
            + example["label"]
            + "\n\n"
        )
    out_prompt = out_prompt + "article: " + input_sample + "\ncategory: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        f"Categorize the following tweet into one of the following categories: "
        f"crime-war-conflict, spiritual, health, politics, human-rights-press-freedom, "
        f"education, business-and-economy, art-and-entertainment, others, "
        f"science-and-technology, sports, environment\n"
        f"Provide only label and in English.\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert tweet annotator and know how to categorize news tweet.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.lower()
    label_fixed = label_fixed.replace("category: ", "")
    label_fixed = label_fixed.replace("text:", "")
    if label_fixed != "true" or label_fixed != "false":
        if len(label_fixed.split()) > 1:
            label_fixed = label_fixed.split()[0]
    else:
        label_fixed = None

    return label_fixed
