from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import AzureModel
from llmebench.tasks import SubjectivityTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                'Classify the tweet as "Objective" or "Subjective". Provide only the classification.\n\n'
                + "Tweet: "
                + input_sample
                + "Classification: "
            ),
        }
    ]


def post_process(response):
    # print(response)
    if "output" in response:
        # if "content" in response["messages"]:
        label = response["output"].strip()
        label = label.replace("<s>", "")
        label = label.replace("</s>", "")
        label = label.lower()
    else:
        print("Response .. " + str(response))
        return None
    if "غير" in label or "subj" in label or "not" in label or "ليس" in label:
        return "SUBJ"
    elif "موضوعية" in label or "obj" in label:
        return "OBJ"
    else:
        return None
