from llmebench.datasets import EmotionDataset
from llmebench.models import FastChatModel
from llmebench.tasks import EmotionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": EmotionDataset,
        "task": EmotionTask,
        "model": FastChatModel,
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
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f"Predict all the possible emotions in the following Arabic sentence without explanation and put them in a Python list. List of emotions is: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, and trust.\n "
        f"Sentence: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
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
        v = 0
        if x.lower() in labels:
            v = 1
        labels_arr.append(v)
    return labels_arr


def post_process(response):
    out = emotions_array(response["choices"][0]["message"]["content"])

    return out
