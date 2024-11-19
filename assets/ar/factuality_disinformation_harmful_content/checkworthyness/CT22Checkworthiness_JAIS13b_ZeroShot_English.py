from llmebench.datasets import CT22CheckworthinessDataset
from llmebench.models import FastChatModel
from llmebench.tasks import CheckworthinessTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }


,
    }


def config():
    return {
        "dataset": CT22CheckworthinessDataset,
        "task": CheckworthinessTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    base_prompt = (
        f'Classify the tweet as "checkworthy" or "not checkworthy". Provide the classification only for the last tweet, without providing any additional justification:\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip()
    
  
 
    label = label.lower()
    if label == "checkworthy" or label == "Checkworthy":
        label_fixed = "1"
    elif label == "Not_checkworthy." or label == "not_checkworthy":
        label_fixed = "0"
    elif "not_checkworthy" in label or "label: not_checkworthy" in label:
        label_fixed = "0"
    elif "checkworthy" in label or "label: checkworthy" in label:
        label_fixed = "1"
    else:
        label_fixed = None

    return label_fixed
