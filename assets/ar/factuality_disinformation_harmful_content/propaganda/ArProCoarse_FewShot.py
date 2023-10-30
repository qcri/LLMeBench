import re

from llmebench.datasets import ArProCoarse
from llmebench.models import OpenAIModel
from llmebench.tasks import MultilabelPropagandaTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
    }


def config():
    return {
        "dataset": ArProCoarse,
        "task": MultilabelPropagandaTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/task1A_test.jsonl"  # os.environ["FILE_PATH"],
        },
    }


def prompt(input_sample, examples):
    prompt_text = (
        f"Your task is to analyze the text and determine if it contains elements of propaganda.\n\n"
        f"The following coarse-grained propaganda techniques is defined based on their appearance of any of the fine-grained propaganda techniques. The left side of the equal sign indicate coarse-grained techniques and right side indicate fine-grained techniques.\n\n"
        f"no_technique = ['no propaganda']\n"
        f"Manipulative Wording = ['Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation, Vagueness, Confusion' , 'Repetition']\n"
        f"Reputation = ['Appeal to Hypocrisy' , 'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation']\n"
        f"Justification = ['Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving']\n"
        f"Simplification = ['Causal Oversimplification' , 'Consequential Oversimplification' , 'False Dilemma/No Choice']\n"
        f"Distraction = ['Red Herring' , 'Straw Man' , 'Whataboutism']\n"
        f"Call = ['Appeal to Time' , 'Conversation Killer' , 'Slogans']\n"
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
        f"Based on the instructions and examples above analyze the following text and provide only labels as a list of string.\n\n"
    )
    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"]  # .lower()
    # pred_label = eval(label)

    labels = []

    response = [
        s.strip().replace("'", "").replace("[", "").replace("]", "")
        for s in label.split(",")
        if len(s) > 1
    ]

    # print(response)
    for label in response:
        label = label.lower()
        if "manipulative" in label:
            labels.append("Manipulative_Wording")
        if "call" in label:
            labels.append("Call")
        if "reputation" in label:
            labels.append("Reputation")
        if "technique" in label or "propaganda" in label:
            labels.append("no_technique")
        if "justification" in label:
            labels.append("Justification")
        if "simplification" in label:
            labels.append("Simplification")
        if "distraction" in label:
            labels.append("Distraction")

    if len(labels) == 0:
        labels.append("no_technique")
    if len(labels) > 1 and "no_technique" in labels:
        labels.remove("no_technique")

    return labels
