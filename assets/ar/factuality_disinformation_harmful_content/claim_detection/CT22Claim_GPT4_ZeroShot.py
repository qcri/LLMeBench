from llmebench.datasets import CT22ClaimDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClaimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.587"},
    }


def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f"Given the following tweet, please identify if it contains a claim. If it does, annotate 'yes', if it does not, annotate 'no'\n\n"
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are a social media expert and a fact-checker. Your role would be to evaluate the provided tweet and determine whether or not it contains a claim. This would involve understanding the content of the tweet and assessing it for statements that could be identified as a claim.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()

    if "label: " in label:
        arr = label.split("label: ")
        label = arr[1].strip()

    if label == "yes" or label == "the sentence contains a factual claim":
        label_fixed = "1"
    if label == "no":
        label_fixed = "0"

    return label_fixed
