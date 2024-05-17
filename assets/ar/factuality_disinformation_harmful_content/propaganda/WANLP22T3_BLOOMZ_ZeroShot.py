import random
import re

from llmebench.datasets import WANLP22T3PropagandaDataset
from llmebench.models import PetalsModel
from llmebench.tasks import MultilabelPropagandaTask


random.seed(1333)


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals.",
        "scores": {"Micro-F1": "0.108"},
    }


def config():
    return {
        "dataset": WANLP22T3PropagandaDataset,
        "dataset_args": {"techniques_path": "classes.txt"},
        "task": MultilabelPropagandaTask,
        "model": PetalsModel,
        "model_args": {
            "class_labels": [
                "no technique",
                "Smears",
                "Exaggeration/Minimisation",
                "Loaded Language",
                "Appeal to fear/prejudice",
                "Name calling/Labeling",
                "Slogans",
                "Repetition",
                "Doubt",
                "Obfuscation, Intentional vagueness, Confusion",
                "Flag-waving",
                "Glittering generalities (Virtue)",
                "Misrepresentation of Someone's Position (Straw Man)",
                "Presenting Irrelevant Data (Red Herring)",
                "Appeal to authority",
                "Whataboutism",
                "Black-and-white Fallacy/Dictatorship",
                "Thought-terminating cliché",
                "Causal Oversimplification",
            ],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return {
        "prompt": (
            f'Label this "tweet" based on the following propaganda techniques:\n\n'
            + f"'no technique' , 'Smears' , 'Exaggeration/Minimisation' , 'Loaded Language' , 'Appeal to fear/prejudice' , 'Name calling/Labeling' , 'Slogans' , 'Repetition' , 'Doubt' , 'Obfuscation, Intentional vagueness, Confusion' , 'Flag-waving' , 'Glittering generalities (Virtue)' , 'Misrepresentation of Someone's Position (Straw Man)' , 'Presenting Irrelevant Data (Red Herring)' , 'Appeal to authority' , 'Whataboutism' , 'Black-and-white Fallacy/Dictatorship' , 'Thought-terminating cliché' , 'Causal Oversimplification'"
            + f"\nGive the list of techniques seperated by a comma. Multiple techniques are allowed: \n"
            + f"tweet: {input_sample}\n\n"
            + f"labels: \n"
        )
    }


def fix_label(pred_label):
    class_labels = [
        "no technique",
        "Smears",
        "Exaggeration/Minimisation",
        "Loaded Language",
        "Appeal to fear/prejudice",
        "Name calling/Labeling",
        "Slogans",
        "Repetition",
        "Doubt",
        "Obfuscation, Intentional vagueness, Confusion",
        "Flag-waving",
        "Glittering generalities (Virtue)",
        "Misrepresentation of Someone's Position (Straw Man)",
        "Presenting Irrelevant Data (Red Herring)",
        "Appeal to authority",
        "Whataboutism",
        "Black-and-white Fallacy/Dictatorship",
        "Thought-terminating cliché",
        "Causal Oversimplification",
    ]
    class_labels = [c.lower() for c in class_labels]

    pred_labels_bool = [bool(re.search(c.lower(), pred_label)) for c in class_labels]
    pred_labels = [class_labels[i].lower() for i, c in enumerate(pred_labels_bool) if c]

    labels_fixed = []

    if len(pred_labels) == 0:
        return ["no technique"]

    for label in pred_labels:
        label = label.replace(".", "").strip()
        label = re.sub("-", " ", label)
        label = label.strip().lower()

        # Handle case of single word labels like "Smears" so we just capitalize it
        label_fixed = label.capitalize()

        if "slogan" in label:
            label_fixed = "Slogans"
        if "loaded" in label:
            label_fixed = "Loaded Language"
        if "prejudice" in label or "fear" in label or "mongering" in label:
            label_fixed = "Appeal to fear/prejudice"
        if "terminating" in label or "thought" in label:
            label_fixed = "Thought-terminating cliché"
        if "calling" in label or label == "name c":
            label_fixed = "Name calling/Labeling"
        if "minimisation" in label or label == "exaggeration minim":
            label_fixed = "Exaggeration/Minimisation"
        if "glittering" in label:
            label_fixed = "Glittering generalities (Virtue)"
        if "flag" in label:
            label_fixed = "Flag-waving"
        if "obfuscation" in label:
            label_fixed = "Obfuscation, Intentional vagueness, Confusion"
        if "oversimplification" in label or "causal" in label:
            label_fixed = "Causal Oversimplification"
        if "authority" in label:
            label_fixed = "Appeal to authority"
        if "dictatorship" in label or "black" in label or "white" in label:
            label_fixed = "Black-and-white Fallacy/Dictatorship"
        if "herring" in label or "irrelevant" in label:
            label_fixed = "Presenting Irrelevant Data (Red Herring)"
        if "straw" in label or "misrepresentation" in label:
            label_fixed = "Misrepresentation of Someone's Position (Straw Man)"
        if "whataboutism" in label:
            label_fixed = "Whataboutism"

        if (
            "no propaganda" in label
            or "technique" in label
            or label == ""
            or label == "no"
            or label == "appeal to history"
            or label == "appeal to emotion"
            or label == "appeal to"
            or label == "appeal"
            or label == "appeal to author"
            or label == "emotional appeal"
            or "no techn" in label
            or "hashtag" in label
            or "theory" in label
            or "specific mention" in label
            or "religious" in label
            or "gratitude" in label
        ):
            label_fixed = "no technique"

        labels_fixed.append(label_fixed)

    out_put_labels = []

    # Remove no technique label when we have other techniques for the same text
    if len(labels_fixed) > 1:
        for flabel in labels_fixed:
            if flabel != "no technique":
                out_put_labels.append(flabel)
        return out_put_labels

    return labels_fixed


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

    pred_label = fix_label(label)

    return pred_label
