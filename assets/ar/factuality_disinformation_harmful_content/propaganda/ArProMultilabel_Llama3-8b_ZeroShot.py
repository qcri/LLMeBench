import ast
import codecs
import random
import re

from llmebench.datasets import ArProMultilabelDataset
from llmebench.models import AzureModel
from llmebench.tasks import MultilabelPropagandaTask


random.seed(1333)

ESCAPE_SEQUENCE_RE = re.compile(
    r"""
    ( \\U........      # 8-digit hex escapes
    | \\u....          # 4-digit hex escapes
    | \\x..            # 2-digit hex escapes
    | \\[0-7]{1,3}     # Octal escapes
    | \\N\{[^}]+\}     # Unicode characters by name
    | \\[\\'"abfnrtv]  # Single-character escapes
    )""",
    re.UNICODE | re.VERBOSE,
)


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": ArProMultilabelDataset,
        "dataset_args": {"techniques_path": "classes.txt"},
        "task": MultilabelPropagandaTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": [
                "Appeal_to_Authority",
                "Appeal_to_Fear-Prejudice",
                "Appeal_to_Hypocrisy",
                "Appeal_to_Popularity",
                "Appeal_to_Time",
                "Appeal_to_Values",
                "Causal_Oversimplification",
                "Consequential_Oversimplification",
                "Conversation_Killer",
                "Doubt",
                "Exaggeration-Minimisation",
                "False_Dilemma-No_Choice",
                "Flag_Waving",
                "Guilt_by_Association",
                "Loaded_Language",
                "Name_Calling-Labeling",
                "Obfuscation-Vagueness-Confusion",
                "Questioning_the_Reputation",
                "Red_Herring",
                "Repetition",
                "Slogans",
                "Straw_Man",
                "Whataboutism",
                "no_technique",
            ],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        f"Your task is to analyze the text and determine if it contains the following propaganda techniques.\n\n"
        f"'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice' , 'no technique'. \nProvide only labels as a list of strings.\n\n"
    )
    out_prompt = prompt_text + "text: " + input_sample + "\nlabel: \n"

    return [
        {
            "role": "user",
            "content": out_prompt,
        }
    ]


def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), "unicode-escape")

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


def fix_single_label(label):
    label_fixed = ""
    if "slogan" in label:
        label_fixed = "Slogans"
    if "loaded" in label:
        label_fixed = "Loaded_Language"
    if "prejudice" in label or "fear" in label or "mongering" in label:
        label_fixed = "Appeal_to_Fear-Prejudice"
    if (
        "terminating" in label
        or "thought" in label
        or "conversation" in label
        or "killer" in label
    ):
        label_fixed = "Conversation_Killer"
    if "calling" in label or label == "name c" or "labeling" in label:
        label_fixed = "Name_Calling-Labeling"
    if (
        "minimisation" in label
        or label == "exaggeration minim"
        or "exaggeration" in label
    ):
        label_fixed = "Exaggeration-Minimisation"
    if "values" in label:
        label_fixed = "Appeal_to_Values"
    if "flag" in label or "wav" in label:
        label_fixed = "Flag_Waving"
    if "obfusc" in label or "vague" in label or "confusion" in label:
        label_fixed = "Obfuscation-Vagueness-Confusion"
    if "causal" in label:
        label_fixed = "Causal_Oversimplification"
    if "conseq" in label:
        label_fixed = "Consequential_Oversimplification"
    if "authority" in label:
        label_fixed = "Appeal_to_Authority"
    if "choice" in label or "dilemma" in label or "false" in label:
        label_fixed = "False_Dilemma-No_Choice"
    if "herring" in label or "irrelevant" in label:
        label_fixed = "Red_Herring"
    if "straw" in label or "misrepresentation" in label:
        label_fixed = "Straw_Man"
    if "guilt" in label or "association" in label:
        label_fixed = "Guilt_by_Association"
    if "questioning" in label or "reputation" in label:
        label_fixed = "Questioning_the_Reputation"
    if "whataboutism" in label:
        label_fixed = "Whataboutism"
    if "doubt" in label:
        label_fixed = "Doubt"
    if "doubt" in label:
        label_fixed = "Doubt"
    if "time" in label:
        label_fixed = "Appeal_to_Time"
    if "popularity" in label:
        label_fixed = "Appeal_to_Popularity"
    if "repetition" in label:
        label_fixed = "Repetition"
    if "hypocrisy" in label:
        label_fixed = "Appeal_to_Hypocrisy"

    if (
        "no propaganda" in label
        or "no technique" in label
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
        or "sarcasm" in label
        or "frustration" in label
        or "analogy" in label
        or "metaphor" in label
        or "religious" in label
        or "gratitude" in label
        or "no_technique" in label
        or "technique" in label
    ):
        label_fixed = "no_technique"

    return label_fixed


def fix_multilabel(pred_label):
    if (
        "used in this text" in pred_label
        or "no technique" in pred_label
        or "[]" in pred_label
        or pred_label == ""
    ):
        return ["no_technique"]

    if "cannot" in pred_label or "not able" in pred_label:
        return None

    labels_fixed = []
    pred_label = (
        pred_label.replace("'label: ", "")
        .replace("'label': ", "")
        .replace('""', '"')
        .replace("''", "'")
    )

    pred_label = decode_escapes(pred_label).replace("'", '"')
    if not pred_label.startswith("["):
        pred_label = "[" + pred_label + "]"
    pred_label = ast.literal_eval(pred_label)

    for label in pred_label:
        label = label.strip().lower()
        label_fixed = fix_single_label(label)
        labels_fixed.append(label_fixed)

    out_put_labels = []
    # Remove no technique label when we have other techniques for the same text
    if len(labels_fixed) > 1:
        for flabel in labels_fixed:
            if flabel != "no_technique":
                out_put_labels.append(flabel)
        return out_put_labels

    return labels_fixed


def post_process(response):
    label = response["output"].strip().lower()
    label = label.replace("<s>", "").replace("</s>", "")
    label = label.lower()

    labels = fix_multilabel(label)

    return labels
