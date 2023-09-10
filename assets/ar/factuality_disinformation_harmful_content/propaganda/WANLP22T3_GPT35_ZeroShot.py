import regex as re

from llmebench.datasets import WANLP22T3PropagandaDataset
from llmebench.models import LegacyOpenAIModel
from llmebench.tasks import MultilabelPropagandaTask


def config():
    return {
        "dataset": WANLP22T3PropagandaDataset,
        "dataset_args": {
            "techniques_path": "data/factuality_disinformation_harmful_content/propaganda/classes.txt"
        },
        "task": MultilabelPropagandaTask,
        "task_args": {},
        "model": LegacyOpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/propaganda/task1_test_gold_label_final.json"
        },
    }


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": "From the text below detect the following propaganda techniques: "
                + "no-technique, smears, exaggeration-minimisation, loaded-language, appeal-to-fear-prejudice, name-calling-labeling, slogans, repetition, doubt, obfuscation-intentional-vagueness-confusion, flag-waving, glittering-generalities-virtue, presenting-irrelevant-data-red-herring, appeal-to-authority, whataboutism, black-and-white-fallacy-dictatorship, thought-terminating-cliché, causal-oversimplification. "
                + "\n Provide one or more labels.\n\n"
                + "sentence: "
                + input_sample
                + "label: \n",
            }
        ],
    }


def fix_label(pred_label):
    if "\n-" in pred_label:
        labels = pred_label.split("\n-")
    else:
        labels = pred_label.split(",")
    labels_fixed = []

    for label in labels:
        label = label.replace(".", "").strip()
        label = re.sub("-", " ", label)
        label = label.replace(",", "").strip()

        # Handle case of single word labels like "Smears" so we just capitalize it
        label_fixed = label.capitalize()

        if "slogan" in label:
            label_fixed = "Slogans"
        if "smear" in label:
            label_fixed = "Smears"
        if "doublt" in label:
            label_fixed = "Doubt"
        if "repetition" in label:
            label_fixed = "Repetition"
        if "loaded" in label or "Loaded" in label:
            label_fixed = "Loaded Language"
        if "prejudice" in label or "fear" in label or "mongering" in label:
            label_fixed = "Appeal to fear/prejudice"
        if "terminating" in label or "thought" in label:
            label_fixed = "Thought-terminating cliché"
        if "calling" in label or label == "name c":
            label_fixed = "Name calling/Labeling"
        if "minimisation" in label or label == "exaggeration minim":
            label_fixed = "Exaggeration/Minimisation"
        if "glittering" in label or "virtue" in label:
            label_fixed = "Glittering generalities (Virtue)"
        if "flag" in label or "waving" in label:
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
        if "technique" in label:
            label_fixed = "no technique"
        if "whataboutism" in label:
            label_fixed = "Whataboutism"

        # Unusual cases
        if (
            label == ""
            or label == " "
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

    return labels_fixed


def post_process(response):
    pred_label = response["choices"][0]["text"]
    pred_label = fix_label(pred_label)

    return pred_label
