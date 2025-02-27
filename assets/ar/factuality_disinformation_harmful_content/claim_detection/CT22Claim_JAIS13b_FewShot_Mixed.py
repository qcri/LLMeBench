from llmebench.datasets import CT22ClaimDataset
from llmebench.models import FastChatModel
from llmebench.tasks import ClaimDetectionTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def prompt(input_sample, few_shot_examples):
    few_shot_text = ""
    for example in few_shot_examples:
        few_shot_text += (
            "التغريدة: "
            + example["input"]
            + "\n"
            + "الإجابة: "
            + ("yes" if example["label"] == "1" else "no")
            + "\n\n"
        )

    return [
        {
            "role": "user",
            "content": (
                "هل تحتوي هذه التغريدة على ادعاء يمكن التحقق منه؟ أجب فقط بـ 'yes' أو 'no'. قدم فقط الإجابة.\n\n"
                + few_shot_text
                + "التغريدة: "
                + input_sample
                + "\n"
                + "الإجابة: "
            ),
        }
    ]


def post_process(response):
    try:
        label = response["choices"][0]["message"]["content"]

        label = label.replace("الإجابة:", "").strip()
        label = label.lower()
        if "هذه التغريدة تحتوي" in label:
            return "1"

        if (
            "لا يمكنني" in label
            or "I cannot" in label
            or "sorry" in label
            or "هذه المحادثة غير مناسبة" in label
        ):
            return None
        if (
            "not a factual claim" in label
            or "لا يوجد" in label
            or "not" in label
            or "لا" in label
        ):
            return "0"

        if "نعم" in label or "yes" in label:
            pred_label = "1"
        elif "لا" in label or "no" in label:
            pred_label = "0"
        else:
            pred_label = ""

        print(f"Predicted Label: {pred_label}")

        return pred_label
    except Exception as e:
        print(f"Error in post-processing: {str(e)}")
        return "0"
