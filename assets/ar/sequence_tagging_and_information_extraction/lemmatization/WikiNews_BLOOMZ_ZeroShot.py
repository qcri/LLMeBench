from llmebench.datasets import WikiNewsLemmatizationDataset
from llmebench.models import PetalsModel
from llmebench.tasks import LemmatizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
    }


def config():
    return {
        "dataset": WikiNewsLemmatizationDataset,
        "task": LemmatizationTask,
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": (
            "for every word in the following Arabic sentence, write only the arabic lemma of the word separated by a single space without explanation.\n\n"
            + "sentence: "
            + input_sample
            + "label: \n"
        )
    }


def post_process(response):
    label = response["outputs"]
    label = label.replace("label:", "")
    label = label.replace("label", "")

    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    if (
        label.startswith("Please provide the Arabic sentence")
        or label.startswith("It seems")
        or "is not" in label
    ):
        label = None
    else:
        # TODO: fix hack to handle prediction failure
        label = (None, label.strip())
    return label
