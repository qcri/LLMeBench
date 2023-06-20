import os

from arabic_llm_benchmark.datasets import EmotionDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import EmotionTask


def config():
    return {
        "dataset": EmotionDataset,
        "dataset_args": {},
        "task": EmotionTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "class_labels": [
                "anger",
                "disgust",
                "fear",
                "joy",
                "love",
                "optimism",
                "pessimism",
                "sadness",
                "surprise",
                "trust",
            ],
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/emotion/test-gold.txt"
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are an AI assistant that is an expert in detecting emotions portrayed in textual data",
        },
        {
            "role": "user",
            "content": f"Predict all the possible emotions in the following Arabic sentence without explanation and put them in a Python list. List of emotions is: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, and trust.\n Sentence: {input_sample}",
        },
    ]


emotions_positions = {
    "anger": 0,
    "anticipation": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "love": 5,
    "optimism": 6,
    "pessimism": 7,
    "sadness": 8,
    "surprise": 9,
    "trust": 10,
}


def emotions_array(labels):
    labels_arr = []
    for x, y in emotions_positions.items():
        print(x)
        v = 0
        if x in labels:
            v = 1
        labels_arr.append(v)
    return labels_arr


def post_process(response):
    out = emotions_array(response["choices"][0]["message"]["content"])
    return out
