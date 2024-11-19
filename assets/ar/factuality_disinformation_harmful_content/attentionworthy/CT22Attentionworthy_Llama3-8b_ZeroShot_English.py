from llmebench.datasets import CT22AttentionworthyDataset
from llmebench.models import AzureModel
from llmebench.tasks import AttentionworthyTask




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT22AttentionworthyDataset,
        "task": AttentionworthyTask,
        "model": AzureModel,
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
            "max_tries": 100,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    prompt_string = (
        f'"You are social media expert. You can annotate important tweets and require attention from journalists, fact-checker, and government entities.'
        f'Annotate "tweet" into one of the following categories: yes_discusses_action_taken, harmful, yes_discusses_cure, yes_asks_question, no_not_interesting, yes_other, yes_blame_authorities, yes_contains_advice, yes_calls_for_action\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        
        {
            "role": "user",
            "content": prompt_string,
        },
    ]

import re

def post_process(response):
    print(response)
    if "output" in response:
        # if "content" in response["messages"]:
        label = response["output"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
    else:
        print("Response .. " + str(response))
        label = ""
    
    label_fixed = label.lower().strip()  # Convert to lowercase and strip leading/trailing whitespace
    
    # Update conditions to match labels without surrounding whitespace
    if label_fixed.startswith("no"):
        label_fixed = "no_not_interesting"
    elif "yes_discusses_covid-19_vaccine_side_effects" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "yes_harmful" in label_fixed:
        label_fixed = "harmful"
    elif label_fixed.startswith("yes"):
        label_fixed = label_fixed.strip()  # Keep the original label if it starts with "yes"
    elif "yes_blame_authoritie" in label_fixed:
        label_fixed = "yes_blame_authoritie"
    elif "yes_discusses_action_taken" in label_fixed:
        label_fixed = "yes_discusses_action_taken"
    elif "harmful" in label_fixed:
        label_fixed = "harmful"
    elif "yes_discusses_cure" in label_fixed:
        label_fixed = "yes_discusses_cure"
    elif "yes_asks_question" in label_fixed:
        label_fixed = "yes_asks_question"
    elif "no_not_interesting" in label_fixed:
        label_fixed = "no_not_interesting"
    elif "yes_other" in label_fixed:
        label_fixed = "yes_other"
    elif "yes_blame_authorities" in label_fixed:
        label_fixed = "yes_blame_authorities"
    elif "yes_contains_advice" in label_fixed:
        label_fixed = "yes_contains_advice"
    elif "yes_calls_for_action" in label_fixed:
        label_fixed = "yes_calls_for_action"
    else:
        label_fixed = None

    return label_fixed
