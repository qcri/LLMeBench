from llmebench.datasets import SpamDataset
from llmebench.models import FastChatModel
from llmebench.tasks import SpamTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": SpamDataset,
        "task": SpamTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
    }





def prompt(input_sample):
    base_prompt = "هل تحتوي التغريدة التالية على محتوى سبام / غير مرغوب فيه / مزعج /إعلان أم لا؟ أجب بـ نعم أو لا بدون الحاجة إلى وصف أو تحليل.\n"
    return [
        {
            "role": "user",
            "content": base_prompt
                + "التغريدة: "
                + input_sample
                + "التصنيف: " 
        }
    ]


def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("التصنيف:", "").strip().lower()
    #print("label", label)
    if "لا" in label or "ليست" in label or "not" in label or "ليس" in label or "no" in label:
        return "__label__NOTADS"
    elif "نعم" in label or "إعلان" in label or "spam" in label or "مزعج" in label or "yes" in label or "مرغوب" in label or "غير" in label:
        return "__label__ADS"
    else:
        return None
