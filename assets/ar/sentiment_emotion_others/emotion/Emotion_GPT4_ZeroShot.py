from llmebench.datasets import EmotionDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import EmotionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Jaccard similarity": "0.373"},
    }


def config():
    return {
        "dataset": EmotionDataset,
        "task": EmotionTask,
        "model": OpenAIModel,
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
        v = 0
        if x in labels:
            v = 1
        labels_arr.append(v)
    return labels_arr


def post_process(response):
    out = emotions_array(response["choices"][0]["message"]["content"])
    return out
