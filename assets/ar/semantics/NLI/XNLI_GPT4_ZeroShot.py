import os

from llmebench.datasets import XNLIDataset
from llmebench.models import GPTChatCompletionModel
from llmebench.tasks import XNLITask


def config():
    return {
        "dataset": XNLIDataset,
        "dataset_args": {},
        "task": XNLITask,
        "task_args": {},
        "model": GPTChatCompletionModel,
        "model_args": {
            "api_type": "azure",
            "api_version": "2023-03-15-preview",
            "api_base": os.environ["AZURE_API_URL"],
            "api_key": os.environ["AZURE_API_KEY"],
            "engine_name": os.environ["ENGINE_NAME"],
            "max_tries": 3,
        },
        "general_args": {"data_path": "data/XNLI/xnli.test.ar.tsv"},
    }


def prompt(input_sample):
    sent1, sent2 = input_sample.split("\t")
    prompt_text = "نقدم لك جملتين تمثلان فرضيتين. مهمتك هي تصنيف الفرضية اللاحقة بالنسبة للفرضية المسبقة تبعاً لواحدة من هذه التصنيفات: صحيح (الفرضية اللاحقة تدل على نفس الفرضية المسبقة)، خطأ (الفرضية اللاحقة تناقض الفرضية المسبقة)، أو غير معروف (حيادي). يجب أن يقتصر ردك على واحدة من هذه التصنيفات: صحيح، خطأ، أو غير معروف."
    prompt_text = (
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
            "role": "system",
            "content": "أنت خبير في فهم اللغة العربية.",
        },
        {
            "role": "user",
            "content": prompt_text,
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
