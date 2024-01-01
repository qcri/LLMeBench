from llmebench.datasets import XNLIDataset
from llmebench.models import FastChatModel
from llmebench.tasks import XNLITask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "JAIS-13b",
        "description": "Locally hosted JAIS-13b-chat model using FastChat.",
        "scores": {"Macro-F1": ""},
    }


def config():
    return {
        "dataset": XNLIDataset,
        "task": XNLITask,
        "model": FastChatModel,
    }


def prompt(input_sample):
    sent1, sent2 = input_sample.split("\t")
    prompt_text = "نقدم لك جملتين تمثلان فرضيتين. مهمتك هي تصنيف الفرضية اللاحقة بالنسبة للفرضية المسبقة تبعاً لواحدة من هذه التصنيفات: صحيح (الفرضية اللاحقة تدل على نفس الفرضية المسبقة)، خطأ (الفرضية اللاحقة تناقض الفرضية المسبقة)، أو غير معروف (حيادي). يجب أن يقتصر ردك على واحدة من هذه التصنيفات: صحيح، خطأ، أو غير معروف."
    base_prompt = (
        prompt_text
        + "\nالفرضية المسبقة: "
        + sent1
        + "\nالفرضية اللاحقة: "
        + sent2
        + "\n"
        + "التصنيف: "
    )

    return [
        {
            "role": "user",
            "content": base_prompt,
        },
    ]


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if "غير معروف" in input_label or "حيادي" in input_label:
        pred_label = "neutral"
    elif "صحيح" in input_label or "تدل" in input_label:
        pred_label = "entailment"
    elif "خطأ" in input_label or "تناقض" in input_label:
        pred_label = "contradiction"
    else:
        print(input_label)
        pred_label = None

    return pred_label
