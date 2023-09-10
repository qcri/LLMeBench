from llmebench.datasets import WikiNewsLemmatizationDataset
from llmebench.models import PetalsModel
from llmebench.tasks import LemmatizationTask


def config():
    return {
        "dataset": WikiNewsLemmatizationDataset,
        "dataset_args": {},
        "task": LemmatizationTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sequence_tagging_ner_pos_etc/lemmatization/WikiNews-26-06-2015-RefLemma.txt"
        },
    }


def prompt(input_sample):
    return {
        "prompt": "for every word in the following Arabic sentence, write only the arabic lemma of the word separated by a single space without explanation.\n\n"
        + "sentence: "
        + input_sample
        + "label: \n"
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
