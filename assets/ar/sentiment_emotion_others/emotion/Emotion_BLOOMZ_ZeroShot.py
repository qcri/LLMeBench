import os

from llmebench.datasets import EmotionDataset
from llmebench.models import PetalsModel
from llmebench.tasks import EmotionTask


def config():
    return {
        "dataset": EmotionDataset,
        "dataset_args": {},
        "task": EmotionTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
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
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/emotion/test-gold.txt"
        },
    }


def prompt(input_sample):
    return {
        "prompt": "Predict all the possible emotions in the following Arabic sentence without explanation and put them in a Python list. List of emotions is: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, and trust.\n\n"
        + "sentence: "
        + input_sample
        + "label: "
    }


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
        v = 0
        if x in labels:
            v = 1
        labels_arr.append(v)
    return labels_arr


def post_process(response):
    label = response["outputs"]
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = emotions_array(label)

    return label
