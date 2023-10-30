import ast
import re

from llmebench.datasets import ArProSpan
from llmebench.models import OpenAIModel
from llmebench.tasks import ArProSpanTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": ArProSpan,
        "task": ArProSpanTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    prompt_text = (
        f"Your task is to analyze the text and determine if it contains the following propaganda techniques.\n\n"
        f"'Appeal to Time' , 'Conversation Killer' , 'Slogans' , 'Red Herring' , 'Straw Man' , 'Whataboutism' , 'Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving' , 'Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation/Vagueness/Confusion' , 'Repetition' , 'Appeal to Hypocrisy' , 'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation' , 'Causal Oversimplification' , 'Consequential Oversimplification' , 'False Dilemma/No Choice' , 'no technique'"
        f"Below you will find a few examples of text with coarse-grained propaganda techniques:\n\n"
    )

    fs_prompt = few_shot_prompt(input_sample, prompt_text, examples)
    return [
        {
            "role": "system",
            "content": "You are an expert annotator.",
        },
        {
            "role": "user",
            "content": fs_prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for index, example in enumerate(examples):
        sent = example["input"]
        tech_str = ""
        for t in example["label"]:
            tech_str += "'" + t + "', "

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "text: "
            + sent
            + "\nlabel: "
            + tech_str
            + "\n\n"
        )

    out_prompt = out_prompt + (
        f"Based on the instructions and examples above analyze the following text answer exactly and only by returning a list of the matching labels from the aforementioned techniques and specify the start position and end position of the text span matching each technique."
        f'Use the following templated and return the results as json string  {{"technique": '
        ' , "text": , "start": , "end": }}\n\n'
    )
    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

    return out_prompt


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
        or "religious" in label
        or "gratitude" in label
    ):
        label_fixed = "no_technique"

    return label_fixed


def fix_multilabel(pred_label):
    if "used in this text" in pred_label or "no technique" in pred_label:
        return ["no_technique"]

    labels_fixed = []
    pred_label = pred_label.replace("'", '"')
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


def fix_span(prediction):
    prediction = prediction.replace("'", '"')
    pred_labels = ast.literal_eval(prediction)

    for label in pred_labels:
        label["technique"] = label["technique"].strip().lower()
        label["technique"] = fix_single_label(label["technique"])

    final_labels = []
    if len(pred_labels) > 1:
        for pred_label in pred_labels:
            if pred_label["technique"] != "no_technique":
                final_labels.append(pred_label)

    return final_labels


def post_process(response):
    label = response["choices"][0]["message"]["content"]  # .lower()
    labels = fix_span(label)

    return labels
