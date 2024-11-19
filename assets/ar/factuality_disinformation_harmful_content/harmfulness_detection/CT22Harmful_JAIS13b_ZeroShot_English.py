from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HarmfulDetectionTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar"},
    }


def prompt(input_sample):
    base_prompt = (
        f"Classify the following tweet as 'harmful' or 'not_harmful'. Provide only label.\n\n"
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
    label = response["choices"][0]["message"]["content"].lower()
    print("label: ",label)    
    if "label: harmful" in label or "is harmful" in label:
        return "1"
    if "لا يمكنني" in label or "ai" in label or "لا يمكن" in label:
        return None
    if "غير ضارة" in label or "لا يحتوي على محتوى ضار" in label or "غير" in label or "ليست" in label or "not" in label:
        return "0"
    
    
    else:
        return "1"
