from llmebench.datasets import ArSarcasmDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm2/testing_data.csv",
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an expert in sarcasm detection.\n\n",
        },
        {
            "role": "user",
            "content": (
                'Predict whether the following "tweet" is sarcastic. Return "yes" if the tweet is sarcastic '
                'and "no" if the tweet is not sarcastic. Provide only label.\n\ntweet: '
                + input_sample
                + "\n"
                "label: \n"
            ),
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"]
    content = content.strip().lower()
    if "yes" in content:
        return "TRUE"
    elif "no" in content:
        return "FALSE"

    return None
