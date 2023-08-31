import os
import random
import re

from llmebench.datasets import PropagandaTweetDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import PropagandaMultilabelTask


random.seed(1333)


def config():
    return {
        "dataset": PropagandaTweetDataset,
        "dataset_args": {
            "techniques_path": "data/factuality_disinformation_harmful_content/propaganda/classes.txt"
        },
        "task": PropagandaMultilabelTask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
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
            "max_tries": 30,
        },
        "general_args": {
            "data_path": "data/factuality_disinformation_harmful_content/propaganda/task1_test_gold_label_final.json",
            "fewshot": {
                "train_data_path": "data/factuality_disinformation_harmful_content/propaganda/task1_train.json",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    out_prompt = out_prompt + "Here are some examples:\n\n"
    for index, example in enumerate(examples):
        tech_str = ""
        for t in example["label"]:
            tech_str += "'" + t + "', "

        out_prompt = (
            out_prompt
            + "Example "
            + str(index)
            + ":"
            + "\n"
            + "tweet: "
            + example["input"]
            + "\nlabel: "
            + tech_str
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        f'Label this "tweet" based on the following propaganda techniques:\n\n'
        f"'no technique' , 'Smears' , 'Exaggeration/Minimisation' , 'Loaded Language' , 'Appeal to fear/prejudice' , 'Name calling/Labeling' , 'Slogans' , 'Repetition' , 'Doubt' , 'Obfuscation, Intentional vagueness, Confusion' , 'Flag-waving' , 'Glittering generalities (Virtue)' , 'Misrepresentation of Someone's Position (Straw Man)' , 'Presenting Irrelevant Data (Red Herring)' , 'Appeal to authority' , 'Whataboutism' , 'Black-and-white Fallacy/Dictatorship' , 'Thought-terminating cliché' , 'Causal Oversimplification'"
        f"Provide only labels as a list of string.\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert social media content analyst.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]  # .lower()
    pred_label = eval(label)

    return pred_label
