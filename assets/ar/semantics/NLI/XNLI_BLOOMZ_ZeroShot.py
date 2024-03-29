from llmebench.datasets import XNLIDataset
from llmebench.models import PetalsModel
from llmebench.tasks import XNLITask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Accuracy": "0.500"},
    }


def config():
    return {
        "dataset": XNLIDataset,
        "task": XNLITask,
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    sent1, sent2 = input_sample.split("\t")

    prompt_text = "You are provided with a premise and a hypothesis. Your task is to classify the hypothesis as either true (entailing the premise), false (contradicting the premise), or unknown (neutral) based on the given premise. The output should only be exactly one of three labels: true, false or unknown."
    prompt_text = (
        prompt_text
        + "\nPremise: "
        + sent1
        + "\nHypothesis: "
        + sent2
        + "\n"
        + "label: "
    )

    return {"prompt": prompt_text}


def post_process(response):
    input_label = (
        response["outputs"].replace("<s>", "").replace("</s>", "").strip().lower()
    )

    if "neutral" in input_label or "unknown" in input_label:
        pred_label = "neutral"
    elif "true" in input_label or "entailment" in input_label:
        pred_label = "entailment"
    elif "false" in input_label or "contradiction" in input_label:
        pred_label = "contradiction"
    else:
        pred_label = None

    return pred_label
