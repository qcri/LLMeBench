from llmebench.datasets import ANSStanceDataset
from llmebench.models import FastChatModel
from llmebench.tasks import StanceTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": ANSStanceDataset,
        "task": StanceTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["agree", "disagree"],
            "max_tries": 30,
        },
    }


def prompt(input_sample):
    ref_s = input_sample["sentence_1"]
    claim = input_sample["sentence_2"]
    base_prompt = (
        f"Given a reference sentence and a claim, predict whether the claim agrees or disagrees with the reference sentence. Reply only using 'agree', 'disagree', or use 'other' if the sentence and claim are unrelated."
        f"\n\n"
        f"reference sentence: {ref_s}"
        f"\nclaim: {claim}"
        f"\nlabel: \n"
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
    return out.lower()
