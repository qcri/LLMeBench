from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HateSpeechTask



def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }



def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["HS", "NOT_HS"],
            "max_tries": 3,
        },
    }

def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                "صنف التغريدة التالية حسب احتوائها على خطاب كراهية. أجب بـ 'HS' أو 'NOT_HS' وقدم التصنيف فقط.\n\n"
                + "التغريدة: "
                + input_sample
                + "\nالتصنيف: "
            ),
        }
    ]


def post_process(response):
    print(response)
    label = response["choices"][0]["message"]["content"].lower()

    if "لا" in label or "ليست" in label or "ليس" in label:
        return "NOT_HS"
    return "HS"
