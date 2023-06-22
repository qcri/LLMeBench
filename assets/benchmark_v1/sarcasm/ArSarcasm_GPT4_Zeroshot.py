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
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm/ArSarcasm_testdata.csv"
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "## INSTRUCTION\nYou are an expert in sarcasm detection.\n\n",
        },
        {
            "role": "user",
            "content": 'You are an AI assistant, an expert at detecting sarcasm in text. Say yes if the tweet is sarcastic and say no if the tweet is not sarcastic: "'
            + input_sample
            + '"',
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].lower()

    if (
        content.startswith("no")
        or "\nNo" in content
        or "tweet is not sarcastic" in content
        or "answer is no" in content
        or "would say no" in content
    ):
        return "FALSE"
    elif content == "yes" or content == "نعم":
        return "TRUE"
    else:
        return None
