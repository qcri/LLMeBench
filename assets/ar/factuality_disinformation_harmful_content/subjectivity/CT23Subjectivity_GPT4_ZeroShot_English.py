from llmebench.datasets import CT23SubjectivityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SubjectivityTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT23SubjectivityDataset,
        "task": SubjectivityTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["SUBJ", "OBJ"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar/dev"},
    }


def prompt(input_sample):
    prompt_string = (
        f'Classify the tweet as "objective" or "subjective". Provide only the label.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert in tweet classification and analysis.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"].strip().lower()
    if "obj" in label or "موضوعي" in label:
        return "OBJ"
    elif "subj" in label or "غير" in label or "لا" in label or "ذاتي" in label or "ليس" in label :
        return "SUBJ"
    return None
