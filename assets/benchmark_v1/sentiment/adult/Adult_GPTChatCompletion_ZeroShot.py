import os

from arabic_llm_benchmark.datasets import AdultDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import AdultTask


def config():
    return {
        "dataset": AdultDataset,
        "dataset_args": {},
        "task": AdultTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/adult/adult-test.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = (
        # f'Classify whether the following Arabic sentence is appropriate for children. If not appropriate, just write "ADULT" without explaination, and if appropriate, just write "NOT_ADULT" without explaination \n {input_sample}'
        f'Given the following tweet, label it as "ADULT" or "NOT_ADULT" based on the content of the tweet.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label adult content within a tweet.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"].replace("label: ", "")
    j = out.find(".")
    if j > 0:
        out = out[0:j]
    return out
