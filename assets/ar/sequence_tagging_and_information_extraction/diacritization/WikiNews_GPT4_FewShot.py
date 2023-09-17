from llmebench.datasets import WikiNewsDiacritizationDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicDiacritizationTask


def config():
    return {
        "dataset": WikiNewsDiacritizationDataset,
        "dataset_args": {},
        "task": ArabicDiacritizationTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"]
        label = example["label"]
        output_prompt = output_prompt + f"Sentence: {tokens}\nLabels: {label}\n"
    output_prompt = output_prompt + f"Sentence: {input_sample}\n" + "Labels:"
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f"Diacritize fully the following Arabic sentence including adding case endings:\n\
                     Make sure to put back non-Arabic tokens intact into the output sentence.\
                    "
    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    text = response["choices"][0]["message"]["content"]

    return text
