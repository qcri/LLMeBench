from llmebench.datasets import ADIDataset
from llmebench.models import PetalsModel
from llmebench.tasks import DialectIDTask


def config():
    return {
        "dataset": ADIDataset,
        "dataset_args": {},
        "task": DialectIDTask,
        "task_args": {},
        "model": PetalsModel,
        "model_args": {
            "class_labels": [
                "EGY",
                "IRA",
                "JOR",
                "KSA",
                "KUW",
                "LEB",
                "LIB",
                "MOR",
                "MSA",
                "PAL",
                "QAT",
                "SUD",
                "SYR",
                "UAE",
                "YEM",
            ],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    arr = input_sample.split()
    if len(arr) > 500:
        input_sample = arr[:500]

    prompt_string = (
        f'Classify the following "text" into one of the following categories: "EGY", "IRA", "JOR", "KSA", "KUW", "LEB", "LIB", "MOR", "MSA", "PAL", "QAT", "SUD", "SYR", "UAE", "YEM"\n'
        f"Please provide only the label.\n\n"
        f"text: {input_sample}\n"
        f"label: \n"
    )

    return {
        "prompt": prompt_string,
    }


def post_process(response):
    label = response["outputs"].strip()
    label = label.replace("<s>", "")
    label = label.replace("</s>", "")
    label = label.lower()

    # label_list = config()["model_args"]["class_labels"]
    # label_list = [lab.lower() for lab in label_list]
    #
    # if "label: " in label:
    #     label_fixed = label.replace("label: ", "").lower()
    # elif label.lower() in label_list:
    #     label_fixed = label.lower()
    # else:
    #     label_fixed = None
    label_list = config()["model_args"]["class_labels"]
    label_list = [dialect.lower() for dialect in label_list]

    label = label.replace("label:", "").strip()

    if label in label_list:
        label_fixed = label
    elif "\n msa" in label:
        label_fixed = "msa"
    elif "\n ksa" in label:
        label_fixed = "ksa"
    elif "\n pal" in label:
        label_fixed = "pal"
    elif "\n egy" in label:
        label_fixed = "egy"
    elif "\n yem" in label:
        label_fixed = "yem"
    elif "\n syr" in label:
        label_fixed = "syr"
    elif "\n jor" in label:
        label_fixed = "jor"
    elif "\n ira" in label:
        label_fixed = "ira"
    elif "\n kuw" in label:
        label_fixed = "kuw"
    else:
        label_fixed = None

    return label_fixed
