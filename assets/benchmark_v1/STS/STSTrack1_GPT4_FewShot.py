import os

from llmebench.datasets import STSArSemEval17Track1Dataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import STSTrack1Task


def config():
    return {
        "dataset": STSArSemEval17Track1Dataset,
        "dataset_args": {},
        "task": STSTrack1Task,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "max_tries": 3,
        },
        "general_args": {
            "data_path": {
                "sentences_path": "data/STS/semeval-2017/STS2017.eval.v1.1/STS.input.track1.ar-ar.txt",
                "gt_data_path": "data/STS/semeval-2017/STS2017.gs/STS.gs.track1.ar-ar.txt",
            },
            "fewshot": {
                "train_data_path": "data/STS/semeval-2017/ar_sts_data_updated/Ar_STS/ar.STS.All.txt",
            },
        },
    }


def prompt(input_sample, examples):
    s1, s2 = input_sample.split("\t")
    base_prompt = (
        "Given two sentences, produce a continuous valued similarity score on a scale "
        "from 0 to 5, with 0 indicating that the semantics of the sentences are "
        "completely independent and 5 signifying semantic equivalence. The output "
        "should be exactly in form Similarity score= ."
    )
    prompt = few_shot_prompt(s1, s2, base_prompt, examples)

    return [
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def few_shot_prompt(s1, s2, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"

    for example in examples:
        ex_s1, ex_s2 = example["input"].split("\t")
        label = example["label"]

        out_prompt = (
            out_prompt
            + "Sentence1: "
            + ex_s1
            + "\nSentence2: "
            + ex_s2
            + "\n"
            + "Similarity score= "
            + str(label)
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = (
        out_prompt
        + "Sentence1: "
        + s1
        + "\nSentence2: "
        + s2
        + "\nSimilarity score= \n"
    )

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.strip().lower()

    if "similarity score=" in input_label:
        pred_label = float(input_label.split("similarity score= ")[1])
    elif input_label.replace(".", "").isnumeric():
        pred_label = float(input_label)
    else:
        print("Issue with predicted score parsing!")
        print(input_label)
        pred_label = None

    return pred_label
