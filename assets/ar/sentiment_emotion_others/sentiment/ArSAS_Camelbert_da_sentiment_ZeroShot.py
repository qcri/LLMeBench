from llmebench.datasets import ArSASDataset
from llmebench.models import HuggingFaceInferenceAPIModel, HuggingFaceTaskTypes
from llmebench.tasks import SentimentTask


def config():
    return {
        "dataset": ArSASDataset,
        "dataset_args": {},
        "task": SentimentTask,
        "task_args": {},
        "model": HuggingFaceInferenceAPIModel,
        "model_args": {
            "task_type": HuggingFaceTaskTypes.Text_Classification,
            "inference_api_url": "https://api-inference.huggingface.co/models/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
            "max_tries": 5,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sentiment/ArSAS-test.txt"
        },
    }


def prompt(input_sample):
    return {"inputs": input_sample}


def post_process(response):
    scores = [(c["label"], c["score"]) for c in response[0]]
    label = sorted(scores, key=lambda x: x[1])[-1][0]
    return label[0].upper() + label[1:].lower()
