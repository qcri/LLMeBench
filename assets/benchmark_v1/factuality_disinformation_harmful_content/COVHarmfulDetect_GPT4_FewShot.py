import os

from arabic_llm_benchmark.datasets import CovidHarmfulDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import HarmfulDetectionTask


def config():
    return {
        "dataset": CovidHarmfulDataset,
        "dataset_args": {},
        "task": HarmfulDetectionTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "max_tries": 3,
        },

        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/harmful/CT22_arabic_1C_harmful_test_gold.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/harmful/CT22_arabic_1C_harmful_train.tsv"
            },
        },
    }


def prompt(input_sample, examples):
    base_prompt = "Classify the sentence as harmful or not harmful. Answer only by yes or no. Provide only label.\n"
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
        input_label.startswith("harmful")
        or input_label.startswith("yes")
        or "label: 1" in input_label
        or "label: yes" in input_label
        or "label: harmful" in input_label
    ):
        pred_label = "1"

    if (
        (
            input_label.startswith("no")
            or input_label == "label: safe"
            or "not harmful" in input_label
        )
        or "label: 0" in input_label
        or "label: no" in input_label
        or "label: not harmful" in input_label
    ):
        pred_label = "0"

    if pred_label == "":
        print(input_label)
        pred_label = None

    return pred_label
