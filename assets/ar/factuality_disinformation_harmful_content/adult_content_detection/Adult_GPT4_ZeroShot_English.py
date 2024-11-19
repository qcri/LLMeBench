from llmebench.datasets import AdultDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import AdultTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": AdultDataset,
        "task": AdultTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["ADULT", "NOT_ADULT"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_string = (
        # f'Classify whether the following Arabic sentence is appropriate for children. If not appropriate, just write "ADULT" without explaination, and if appropriate, just write "NOT_ADULT" without explaination \n {input_sample}'
        f'Does the following tweet contain adult content? label it as "ADULT" if it does, or "NOT_ADULT" otherwise. Return label only.\n\n'
        f"tweet: {input_sample}\n"
        f"label: \n"
    )
    return [
        {
            "role": "system",
            "content": "You are an expert annotator, you can identify and label adult content within a tweet.",
        },
        {
            "role": "user",
            "content": prompt_string,
        },
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label = label.replace("التصنيف:", "").strip()
    label = label.replace("label:", "").strip()

    label = label.replace("<s>", "").replace("</s>", "")
    label = label.lower()

    if "ليس" in label or "ليست" in label or "not" in label or "no" in label or "غير" in label or "لا" in label or "not_adult" in label or "not adult" in label:
        return "NOT_ADULT"
    elif "للبالغين" in label or "نعم" in label or "adult" in label:
        return "ADULT"
    else:
        return None
