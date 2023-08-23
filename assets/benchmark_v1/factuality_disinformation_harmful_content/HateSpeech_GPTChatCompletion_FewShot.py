import os

from llmebench.datasets import HateSpeechDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import HateSpeechTask


def config():
    return {
        "dataset": HateSpeechDataset,
        "dataset_args": {},
        "task": HateSpeechTask,
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
            "data_path": "data/factuality_disinformation_harmful_content/hate_speech/OSACT2020-sharedTask-test-tweets-labels.txt",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/hate_speech/OSACT2020-sharedTask-train_HS.txt",  # TO_DO
            },
        },
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Given a tweet, predict whether it contains hate speech. Answer only by using"
        " hate_speech and not_hate_speech. Here are some examples:\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label hate speech content within a tweet.",
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
        label = "not_hate_speech" if example["label"] == "NOT_HS" else "hate_speech"
        out_prompt = (
            out_prompt + "Tweet: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Tweet: " + input_sample + "\nLabel:\n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]

    if "not_hate_speech" in out or "no_hate_speech" in out:
        out = "NOT_HS"
    elif "hate_speech" in out:
        out = "HS"
    else:
        out = None
    return out
