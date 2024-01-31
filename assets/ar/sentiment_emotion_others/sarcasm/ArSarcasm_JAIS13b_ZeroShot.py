from llmebench.datasets import ArSarcasmDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SarcasmTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": ArSarcasmDataset,
        "task": SarcasmTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f'Predict whether the following "tweet" is sarcastic. Return "yes" if the tweet is sarcastic '
        f'and "no" if the tweet is not sarcastic. Provide only label.\n\ntweet: {input_sample} \n'
        f"label: \n"
    )

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    out = out.strip().lower()

    if "i apologize" in out:
        return None

    j = out.find("label:")
    if j > 0:
        out = out[j + len("label:") :]
    else:
        j = out.find(" is:\n\n")
        if j > 0:
            out = out[j + len(" is:\n\n") :]
    out = out.strip().title()
    if out.lower() == "yes":
        return "TRUE"
    elif out.lower() == "no":
        return "FALSE"
    return None
