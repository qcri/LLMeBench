from llmebench.datasets import ASNDDataset
from llmebench.models import FastChatModel
from llmebench.tasks import NewsCategorizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "Jais-13b",
        "description": "Locally hosted Jais-13b-chat model using FastChat.",
        "scores": {"Macro-F1": "0.1955"},
    }


def config():
    return {
        "dataset": ASNDDataset,
        "task": NewsCategorizationTask,
        "model": FastChatModel,
    }


def prompt(input_sample):
    base_prompt = (
        f"Classify the following tweet into one of the following categories: "
        f"spiritual, crime-war-conflict, health, politics, human-rights-press-freedom, "
        f"education, business-and-economy, art-and-entertainment, others, "
        f"science-and-technology, sports or environment\n"
        f"\ntweet: {input_sample}"
        f"\ncategory: \n"
    )

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()

    if "crime-war-conflict" in label or "war" in label:
        label_fixed = "crime-war-conflict"
    elif "spiritual" in label:
        label_fixed = "spiritual"
    elif "health" in label:
        label_fixed = "health"
    elif "politics" in label:
        label_fixed = "politics"
    elif "human-rights-press-freedom" in label:
        label_fixed = "human-rights-press-freedom"
    elif "education" in label:
        label_fixed = "education"
    elif "business-and-economy" in label:
        label_fixed = "business-and-economy"
    elif "art-and-entertainment" in label or "entertainment" in label:
        label_fixed = "art-and-entertainment"
    elif "others" in label:
        label_fixed = "others"
    elif "science-and-technology" in label or "science" in label:
        label_fixed = "science-and-technology"
    elif "sports" in label:
        label_fixed = "sports"
    elif "environment" in label:
        label_fixed = "environment"
    else:
        label_fixed = None


    return label_fixed
