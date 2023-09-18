from llmebench.datasets import CT22ClaimDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClaimDetectionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Accuracy": "0.686"},
    }


def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def prompt(input_sample, examples):
    base_prompt = "Does this sentence contain a factual claim? Answer only by yes or no. Provide only label.\n"
    prompt = few_shot_prompt(input_sample, base_prompt, examples)

    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        label = "no" if example["label"] == "0" else "yes"
        out_prompt = (
            out_prompt + "Sentence: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Sentence: " + input_sample + "\nLabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()
    pred_label = ""

    if (
        "yes" in input_label
        or "contains a factual claim" in input_label
        or "label: 1" in input_label
    ):
        pred_label = "1"
    if (
        input_label == "no"
        or "label: 0" in input_label
        or "label: no" in input_label
        or "not contain a factual claim" in input_label
        or "doesn't contain a factual claim" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        pred_label = None

    return pred_label
