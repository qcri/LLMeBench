from llmebench.datasets import ArProCoarseDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import MultilabelPropagandaTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Micro-F1": "0.540"},
    }


def config():
    return {
        "dataset": ArProCoarseDataset,
        "task": MultilabelPropagandaTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        f"Your task is to analyze the text and determine if it contains elements of propaganda.\n\n"
        f"The following coarse-grained propaganda techniques are defined based on the appearance of any of the fine-grained propaganda techniques. The left side of the equal sign indicates coarse-grained techniques and right side indicates fine-grained techniques.\n\n"
        f"no_technique = ['no propaganda']\n"
        f"Manipulative Wording = ['Exaggeration/Minimisation' , 'Loaded Language' , 'Obfuscation, Vagueness, Confusion' , 'Repetition']\n"
        f"Reputation = ['Appeal to Hypocrisy' , 'Doubt' , 'Guilt by Association' , 'Name Calling/Labeling' , 'Questioning the Reputation']\n"
        f"Justification = ['Appeal to Authority' , 'Appeal to Fear/Prejudice' , 'Appeal to Popularity' , 'Appeal to Values' , 'Flag Waving']\n"
        f"Simplification = ['Causal Oversimplification' , 'Consequential Oversimplification' , 'False Dilemma/No Choice']\n"
        f"Distraction = ['Red Herring' , 'Straw Man' , 'Whataboutism']\n"
        f"Call = ['Appeal to Time' , 'Conversation Killer' , 'Slogans']\n"
    )
    out_prompt = prompt_text + (
        f"Based on the instructions above analyze the following text and provide only coarse-grained propaganda techniques as a list of strings.\n\n"
    )
    out_prompt = out_prompt + "text: " + input_sample + "\nlabel: \n"

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
