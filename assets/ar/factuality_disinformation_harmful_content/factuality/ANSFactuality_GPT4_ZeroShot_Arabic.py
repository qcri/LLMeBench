from llmebench.datasets import ANSFactualityDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import FactualityTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": ANSFactualityDataset,
        "task": FactualityTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    prompt_text = (
        "حدد ما إذا كانت المعلومات في الجملة صحيحة أم خاطئة. "
        "أجب فقط بـ 'صحيحة' أو 'خاطئة'.\n\n"
        + "الجملة: "
        + input_sample
        + "\nالتصنيف: \n"
    )

    return [
        {
            "role": "system",
            "content": "أنت خبير في التحقق من الحقائق.",
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if (
        "true" in input_label
        or "yes" in input_label
        or "right" in input_label
        or "صحيح" in input_label
        or "صحيحة" in input_label
        or "نعم" in input_label
    ):
        return "true"

    elif (
        "false" in input_label
        or "wrong" in input_label
        or "خطأ" in input_label
        or "لا" in input_label
        or "not" in input_label
        or "خاطئة" in input_label
        or "خاطئ" in input_label
    ):
        return "false"

    else:
        return None
