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
            "max_tries": 30,
        },
        "general_args": {
            "fewshot": {
                "deduplicate": True,
            },
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n\n"
    out_prompt = out_prompt + "اليك بعض الامثلة:\n\n"
    for index, example in enumerate(examples):
        label = "للبالغين" if example["label"] == "ADULT" else "ليست للبالغين"

        out_prompt = (
            out_prompt
            + "مثال "
            + str(index)
            + ":"
            + "\n"
            + "التغريدة: "
            + example["input"]
            + "\التصنيف: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "التغريدة: " + input_sample + "\التصنيف: \n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = "هل تحتوي التغريدة التالية على لغة للبالغين؟ اجب 'للبالغين' أو 'ليست للبالغين' بدون أي تفسير."
    return [
        {
            "role": "system",
            "content": "أنت محلل خبير، يمكنك تحديد وتصنيف المحتوى الخاص بالبالغين في التغريدة.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    if not response:
        return None

    label = response["choices"][0]["message"]["content"]
    label = label.replace("التصنيف:", "").strip()
    label = label.replace("label:", "").strip()
    label = label.lower()

    if "ليس" in label or "ليست" in label or "not" in label or "no" in label or "غير" in label or "لا" in label or "not_adult" in label or "not adult" in label:
        return "NOT_ADULT"
    elif "للبالغين" in label or "نعم" in label or "adult" in label:
        return "ADULT"
    else:
        return None
