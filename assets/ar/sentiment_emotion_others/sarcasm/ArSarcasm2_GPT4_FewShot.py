from llmebench.datasets import ArSarcasmDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SarcasmTask


def config():
    return {
        "dataset": ArSarcasmDataset,
        "dataset_args": {},
        "task": SarcasmTask,
        "task_args": {},
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["TRUE", "FALSE"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm2/testing_data.csv",
            "fewshot": {
                "train_data_path": "data/sentiment_emotion_others/sarcasm/ArSarcasm2/training_data.csv",
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    for example in examples:
        label = "not_sarcastic" if example["label"] == "FALSE" else "sarcastic"
        out_prompt = (
            out_prompt + "tweet: " + example["input"] + "\nlabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = 'Predict whether the following "tweet" is sarcastic. Return sarcastic if the tweet is sarcastic and not_sarcastic if the tweet is not sarcastic. Provide only label.'
    return [
        {
            "role": "system",
            "content": f"You are an expert in sarcasm detection.\n",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    content = response["choices"][0]["message"]["content"].lower()

    if "not_sarcastic" in content:
        return "FALSE"
    elif "sarcastic" in content:
        return "TRUE"
    else:
        return None
