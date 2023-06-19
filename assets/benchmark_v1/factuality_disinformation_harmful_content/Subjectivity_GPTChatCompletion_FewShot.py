import os

from arabic_llm_benchmark.datasets import SubjectivityDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import SubjectivityTask


def config():
    return {
        "dataset": SubjectivityDataset,
        "dataset_args": {},
        "task": SubjectivityTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/subjectivity/dev_ar.tsv",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/subjectivity/train_ar.tsv"
            },
        },
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Classify the sentence as Subjective or Objective. Provide only label.\n"
    )
    return [
        {
            "role": "system",
            # "content": "You are social media expert. You can annotate important tweets and require attention from journalists, fact-checker, and government entities.",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]
    # return {
    #     "system_message": "You are an AI assistant that helps people find information.",
    #     "messages": [
    #         {
    #             "sender": "user",
    #             "text": few_shot_prompt(input_sample, base_prompt, examples),
    #         }
    #     ],
    # }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        label = "objective" if example["label"] == "OBJ" else "subjective"
        out_prompt = (
            out_prompt + "Sentence: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Sentence: " + input_sample + "\nLabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if label == "objective" or label == "objective.":
        label_fixed = "OBJ"

    elif label == "subjective" or label == "subjective.":
        label_fixed = "SUBJ"

    return label_fixed
