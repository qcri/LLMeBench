from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SubjectivityTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": FastChatModel,
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    base_prompt = f'صنف التغريدة الى "موضوعية" أو "غير موضوعية". قم بتقديم التصنيف فقط.\n\nالتغريدة: {input_sample}\nالتصنيف: '
    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    label = (
        response["choices"][0]["message"]["content"].lower().replace(".", "").strip()

    )

    if "غير" in label or "subj" in label or "not" in label or "ليس" in label:
        return "SUBJ"
    elif "موضوعية" in label or "obj" in label:
        return "OBJ"
    else:
        return None
