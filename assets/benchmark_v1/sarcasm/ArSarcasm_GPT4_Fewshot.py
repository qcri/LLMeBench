import os

from arabic_llm_benchmark.datasets import ArSarcasmDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 20,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm/ArSarcasm_testdata.csv",
            "fewshot": {
                "train_data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm_Train/ArSarcasm_traindata.csv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"

    for index, example in enumerate(examples):
        label = "not_sarcastic" if example["label"] == False else "sarcastic"
        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + label
            + "\n\n"
        )

    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = 'Annotate the "tweet" into "one" of the following categories: sarcastic or not_sarcastic'
    return [
        {
            "role": "system",
            "content": f"As an AI system, your role is to analyze tweets and classify them as 'sarcastic' or 'not_sarcastic'. Provide only label and in English.\n",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].lower()

    if "not_sarcastic" in content:
        return "FALSE"
    elif "sarcastic" in content:
        return "TRUE"
    else:
        return None
