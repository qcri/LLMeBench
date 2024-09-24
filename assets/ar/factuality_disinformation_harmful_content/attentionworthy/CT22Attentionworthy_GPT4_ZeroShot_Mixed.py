from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AttentionworthyTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Weighted-F1": "0.257"},
    }


def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": [
                "yes_discusses_action_taken",
                "harmful",
                "yes_discusses_cure",
                "yes_asks_question",
                "no_not_interesting",
                "yes_other",
                "yes_blame_authorities",
                "yes_contains_advice",
                "yes_calls_for_action",
            ],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f'هل تتطلب "التغريدة" انتباه الصحفيين، ومدققي الحقائق، والجهات الحكومية؟ قم بتصنيف "التغريدة" التالية إلى واحدة من الفئات التالية: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action. قدم التصنيف فقط.\n\n'
        f"التغريدة: {input_sample}\n"
        f"التصنيف: \n"
    )
    return [
        {
            "role": "system",
            "content": "أنت خبير في وسائل التواصل الاجتماعي. يمكنك تحليل و تصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = (
        label.replace(" - ", ", ")
        .replace(",", "")
        .replace(".", "")
        .replace("label:", "")
    )

    label = (
        label.lower().strip()
    )  # Convert to lowercase and strip leading/trailing whitespace

    if label.startswith("no"):
        label_fixed = "no_not_interesting"
    elif "yes_discusses_covid-19_vaccine_side_effects" in label:
        label_fixed = "yes_discusses_cure"
    elif "yes_harmful" in label:
        label_fixed = "harmful"
    elif "yes_blame_authoritie" in label:
        label_fixed = "yes_blame_authoritie"
    elif "yes_discusses_action_taken" in label:
        label_fixed = "yes_discusses_action_taken"
    elif "harmful" in label:
        label_fixed = "harmful"
    elif "yes_discusses_cure" in label:
        label_fixed = "yes_discusses_cure"
    elif "yes_asks_question" in label:
        label_fixed = "yes_asks_question"
    elif "no_not_interesting" in label:
        label_fixed = "no_not_interesting"
    elif "yes_other" in label:
        label_fixed = "yes_other"
    elif "yes_blame_authorities" in label:
        label_fixed = "yes_blame_authorities"
    elif "yes_contains_advice" in label:
        label_fixed = "yes_contains_advice"
    elif "yes_calls_for_action" in label:
        label_fixed = "yes_calls_for_action"
    elif label.startswith("yes"):
        label_fixed = label
    else:
        label_fixed = None

    return label_fixed
