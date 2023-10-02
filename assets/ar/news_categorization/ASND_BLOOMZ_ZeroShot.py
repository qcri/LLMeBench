from llmebench.datasets import ASNDDataset
from llmebench.models import PetalsModel
from llmebench.tasks import NewsCategorizationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "bloomz-176b (8bit quantized)",
        "description": "Locally hosted BLOOMZ 176b model (8 bit quantized version) using the Petals. Improved prompt with length limit removed from previous version.",
        "scores": {"Macro-F1": "0.371"},
    }


def config():
    return {
        "dataset": ASNDDataset,
        "task": NewsCategorizationTask,
        "model": PetalsModel,
    }


def prompt(input_sample):
    prompt_string = (
        f"You are an expert news editor and know how to categorize news tweets.\n\n"
        f"Categorize the following tweet into one of the following categories: "
        f"crime-war-conflict, spiritual, health, politics, human-rights-press-freedom, "
        f"education, business-and-economy, art-and-entertainment, others, "
        f"science-and-technology, sports, environment\n"
        f"Provide only label and in English.\n\n"
        f"\ntweet: {input_sample}"
        f"\ncategory: \n"
    )

    return {"prompt": prompt_string}


def post_process(response):
    label = response["outputs"].strip().lower()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")

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
