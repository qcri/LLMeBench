import os

from arabic_llm_benchmark.datasets import BnSentimentDataset
from arabic_llm_benchmark.models import BLOOMPetalModel
from arabic_llm_benchmark.tasks import BnSentimentTask


def config():
    return {
        "dataset": BnSentimentDataset,
        "dataset_args": {},
        "task": BnSentimentTask,
        "task_args": {},
        "model": BLOOMPetalModel,
        "model_args": {
            "api_url": os.environ["API_URL"],
            "class_labels": ["Positive", "Negative", "Neutral"],
            "max_tries": 10,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/bn/bn_all_test.tsv"
        },
    }


def prompt(input_sample):
    prompt_string = f"""Label the following text as Neutral Positive, or Negative. Provide only the label as your response. 

        text: {input_sample}

        label: """

    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "").replace("</s>", "").strip()

    return label
