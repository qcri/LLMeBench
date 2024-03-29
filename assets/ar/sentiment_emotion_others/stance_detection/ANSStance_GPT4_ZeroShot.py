from llmebench.datasets import ANSStanceDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import StanceTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Macro-F1": "0.762"},
    }


def config():
    return {
        "dataset": ANSStanceDataset,
        "task": StanceTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["agree", "disagree"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    ref_s = input_sample["sentence_1"]
    claim = input_sample["sentence_2"]
    prompt_string = (
        f"Given a reference sentence and a claim, predict whether the claim agrees or disagrees with the reference sentence. Reply only using 'agree', 'disagree', or use 'other' if the sentence and claim are unrelated."
        f"\n\n"
        f"reference sentence: {ref_s}"
        f"\nclaim: {claim}"
        f"\nlabel: \n"
    )

    return [
        {
            "role": "system",
            "content": "You are a fact checking expert.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    label = label.replace("label:", "").strip()

    label_fixed = None

    if "unrelated" in label or "other" in label:
        label_fixed = "other"
    elif "disagree" in label:
        label_fixed = "disagree"
    elif label == "agree":
        label_fixed = "agree"

    return label_fixed
