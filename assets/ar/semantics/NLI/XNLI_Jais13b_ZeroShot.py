from llmebench.datasets import XNLIDataset
from llmebench.models import FastChatModel
from llmebench.tasks import XNLITask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
        "scores": {"Macro-F1": "0.425"},
    }


def config():
    return {
        "dataset": XNLIDataset,
        "task": XNLITask,
        "model": FastChatModel,
    }


def prompt(input_sample):
    sent1, sent2 = input_sample.split("\t")

    base_prompt = "You are provided with a premise and a hypothesis. Your task is to classify the hypothesis as either true (entailment), false (contradiction), or unknown (neutral) based on the given premise. The output should only be exactly one of three labels: true, false or unknown."
    base_prompt = base_prompt + "\nPremise: " + sent1 + "\nHypothesis: " + sent2 + "\n"

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if "neutral" in input_label or "unknown" in input_label:
        pred_label = "neutral"
    elif "true" in input_label or "entailment" in input_label:
        pred_label = "entailment"
    elif "false" in input_label or "contradiction" in input_label:
        pred_label = "contradiction"
    else:
        pred_label = None

    return pred_label
