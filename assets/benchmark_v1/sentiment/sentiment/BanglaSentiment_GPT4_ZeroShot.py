import os

from arabic_llm_benchmark.datasets import BanglaSentimentDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import SentimentTask


def config():
    return {
        "dataset": BanglaSentimentDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["Positive", "Negative", "Neutral"],
            "max_tries": 20,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/bn/bn_all_test.tsv",
        },
    }


def prompt(input_sample):
    prompt_string = f"""Based on the content of the text, please classify it as either "Positive", "Negative", or "Neutral". Provide only the label as your response. 

        text: {input_sample}

        label: """

    return [
        {
            "role": "system",
            "content": "You are a expert annotator. Your task is to analyze the text and identify sentiment polarity.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    if not response:
        return None
    label = response["choices"][0]["message"]["content"]

    label_fixed = label.replace("label:", "").strip()
    if label_fixed.startswith("Please provide the text"):
        label_fixed = None

    return label_fixed
