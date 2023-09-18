from llmebench.datasets import EmotionDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import EmotionTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Jaccard similarity": "0.489"},
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


def prompt(input_sample, examples):
    base_prompt = "Predict all the possible emotions in the following Arabic tweet without explanation and put them in a Python list. List of emotions is: anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, and trust."

    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.You are an AI assistant that is an expert in detecting emotions portrayed in textual data.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        label_list = ", ".join(map(str, example["label"]))
        out_prompt = (
            out_prompt + "Sentence: " + example["input"] + "\n" + label_list + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Sentence: " + input_sample + "\n"

    return out_prompt


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
