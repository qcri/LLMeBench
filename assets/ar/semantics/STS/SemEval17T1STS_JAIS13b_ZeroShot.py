from llmebench.datasets import SemEval17T1STSDataset
from llmebench.models import FastChatModel
from llmebench.tasks import STSTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b-chat",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
    }


def config():
    return {
        "dataset": SemEval17T1STSDataset,
        "task": STSTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt = (
        f"Given two sentences, produce a continuous valued similarity score on a "
        f"scale from 0 to 5, with 0 indicating that the semantics of the sentences are "
        f"completely independent and 5 signifying semantic equivalence. The output "
        f"should be exactly in form Similarity score =. \n{input_sample}"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    raw_response = response["choices"][0]["message"]["content"]

    if "Similarity score =" in raw_response:
        pred_num = (
            raw_response.split("Similarity score = ")[1]
            .strip()
            .split(" ")[0]
            .rstrip(".")
        )
        score = float(pred_num)
    else:
        try:
            pred_sum = float(raw_response)
            score = pred_sum
        except Exception as e:
            score = None

    return score
