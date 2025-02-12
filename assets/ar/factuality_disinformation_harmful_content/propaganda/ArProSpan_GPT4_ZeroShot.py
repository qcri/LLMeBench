import ast
import codecs
import json
import re

from llmebench.datasets import ArProSpanDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArProSpanTask

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
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. We implemented correcting predicted span by GPT.",
        "scores": {"Micro-F1": "0.117"},
    }


def config():
    return {
        "dataset": ArProSpanDataset,
        "task": ArProSpanTask,
        "task_args": {"correct_span": True},
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        f"Label the text by the following propaganda techniques:\n"
        f"'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , "
        f"'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , "
        f"'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , "
        f"'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , "
        f"'Consequential Oversimplification' , 'False Dilemma/No Choice' , 'no technique'"
    )

    out_prompt = prompt_text + (
        f"\nAnswer exactly and only by returning a list of the matching labels from the aforementioned "
        f"techniques and specify the start position and end position of the text span matching each technique. \n"
        f'Use this template [{{"technique": ,"text": ,"start": ,"end": ,"explanation": }}]\n\n'
    )

    out_prompt = out_prompt + "Paragraph: " + input_sample + "\n\nLabels: \n\n"

    return [
        {
            "role": "system",
            "content": "You are an expert annotator.",
        },
        {
            "role": "user",
            "content": out_prompt,
        },
    ]


def decode_escapes(s):
    def decode_match(match):
        return codecs.decode(match.group(0), "unicode-escape")

    return ESCAPE_SEQUENCE_RE.sub(decode_match, s)


def fix_single_label(label):
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
        or "rhetorical" in label
    ):
        label_fixed = "no_technique"

    return label_fixed


def fix_span(prediction):
    # print(prediction)
    prediction = (
        prediction.replace("},\n{", "}, {")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("[  ", "[")
        .replace("[ ", "[")
        .replace("  {", "{")
        .replace(" ]", "]")
        .replace("  ]", "]")
        .strip()
    )

    # print(prediction)

    if "provide the paragraph" in prediction:
        return []

    try:
        pred_labels = ast.literal_eval(prediction)
    except:
        # print("ERRORRR!")
        pred_labels = json.loads(prediction)

    # print(pred_labels)

    # print(prediction)
    format_pred_label = []
    for i, label in enumerate(pred_labels):
        if (
            "technique" not in label
            or "start" not in label
            or "end" not in label
            or "text" not in label
            or len(label["text"]) < 2
        ):
            continue

        label["technique"] = label["technique"].strip().lower()
        label["technique"] = fix_single_label(label["technique"])

        format_pred_label.append(label)

    if len(format_pred_label) == 0:
        return []

    final_labels = []
    for pred_label in format_pred_label:
        if pred_label["technique"] != "no_technique":
            final_labels.append(pred_label)

    return final_labels


def post_process(response):
    labels = response["choices"][0]["message"]["content"].lower()
    # labels1,labels2 = labels.split("final labels:")
    # labels1 = labels1.replace('labels:','').split("\n")[0].strip()
    # labels1 = fix_span(labels1)
    # labels = fix_span(labels2)

    labels = labels.replace("labels:", "")
    labels = fix_span(labels)

    # if labels1 != labels:
    #     print(labels1)
    #     print('=' * 35)
    #     print(labels)
    # else:
    #     print("=================LABELS BEFORE MATCH AFTER===================")

    return labels
