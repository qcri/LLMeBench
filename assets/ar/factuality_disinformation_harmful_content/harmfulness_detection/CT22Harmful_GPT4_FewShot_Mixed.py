from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HarmfulDetectionTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def prompt(input_sample, examples):
    base_prompt = "هل قد تتسبب هذه التغريدة بأي ضرر أو أذى لشخص أو مؤسسة؟ إن كانت كذلك، أجب بـ 'harmful'. أجب بـ 'not harmful' إن لم تكن كذلك. قدم فقط التصنيف.\n"
    prompt = few_shot_prompt(input_sample, base_prompt, examples)

    return [
        {
            "role": "system",
            "content": "أنت خبير في تحليل وتصنيف التغريدات.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        label = "not harmful" if example["label"] == "0" else "harmful"
        out_prompt = (
            out_prompt + "التغريدة: " + example["input"] + "التصنيف: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "التغريدة: " + input_sample + "التصنيف: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label_fixed = None
    label = label.replace("label:", "")
    label = label.replace("التصنيف:", "")
    label = label.strip().lower()

    if (
        "غير ضارة" in label
        or "ليست ضارة" in label
        or "غير ضاره" in label
        or "غير" in label
        or "not" in label
        or "ليست" in label
        or "لا" in label
        or "not harmful" in label
        or label.startswith("no")
        or "safe" in label
        or "not_harmful" in label
    ):
        return "0"
    elif "ضارة" in label or "harmful" in label or "نعم" in label or "yes" in label:
        return "1"

    return label_fixed
