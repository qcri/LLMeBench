from llmebench.datasets import SpamDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import SpamTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": SpamDataset,
        "task": SpamTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["__label__ADS", "__label__NOTADS"],
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    base_prompt =  "هل تحتوي التغريدة التالية على محتوى سبام / غير مرغوب فيه / مزعج /إعلان أم لا؟ أجب بـ 'إعلان' أو 'ليس إعلان'، قدم التصنيف فقط بدون الحاجة إلى وصف أو تحليل.\n"
    base_prompt += "\n" + "التغريدة: " + input_sample + "\n\nالتصنيف: "
    
    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل و تصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": base_prompt
        },
    ]




def post_process(response):
    out = response["choices"][0]["message"]["content"]
    label = out.replace("التصنيف:", "").strip().lower()
    label = label.replace("label:", "").strip().lower()

    #print("label", label)
    if "لا أستطيع" in label or "I cannot" in label:
        return None
    if  "ليست" in label or "not" in label or "no" in label or "ليس" in label or "notads" in label:
        return "__label__NOTADS"
    elif "نعم" in label or "إعلان" in label or "spam" in label or "مزعج" in label or "اعلان" in label or "مرغوب" in label or "غير" in label or "__ads" in label:
        return "__label__ADS"
    else:
        return None