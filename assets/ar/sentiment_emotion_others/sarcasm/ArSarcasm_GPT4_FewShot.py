from llmebench.datasets import ArSarcasmDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SarcasmTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"F1 (POS)": "0.504"},
    }


def config():
    return {
        "dataset": ArSarcasmDataset,
        "task": SarcasmTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    for example in examples:
        label = "not_sarcastic" if example["label"] == "FALSE" else "sarcastic"
        out_prompt = (
            out_prompt + "tweet: " + example["input"] + "\nlabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = 'Predict whether the following "tweet" is sarcastic. Return sarcastic if the tweet is sarcastic and not_sarcastic if the tweet is not sarcastic. Provide only label.'
    return [
        {
            "role": "system",
            "content": f"You are an expert in sarcasm detection.\n",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].lower()

    if "not_sarcastic" in content:
        return "FALSE"
    elif "sarcastic" in content:
        return "TRUE"
    else:
        return None
