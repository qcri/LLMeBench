from llmebench.datasets import SemEval17T2STSDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import STSTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"PC": "0.857"},
    }


def config():
    return {
        "dataset": SemEval17T2STSDataset,
        "task": STSTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
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
