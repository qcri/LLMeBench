from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Macro-F1": "0.644"},
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Given a tweet, predict whether it contains hate speech. Answer only by using"
        " hate_speech and not_hate_speech. Here are some examples:\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label hate speech content within a tweet.",
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
        # label = "no" if example["label"] == "0" else "yes"
        label = "not_hate_speech" if example["label"] == "NOT_HS" else "hate_speech"
        out_prompt = (
            out_prompt + "Tweet: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Tweet: " + input_sample + "\nLabel:\n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    j = out.find(".")
    if j > 0:
        out = out[0:j]

    if "not_hate_speech" in out or "no_hate_speech" in out:
        out = "NOT_HS"
    elif "hate_speech" in out:
        out = "HS"
    else:
        out = None
    return out
