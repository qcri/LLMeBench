from llmebench.datasets import OSACT4SubtaskADataset
from llmebench.models import FastChatModel
from llmebench.tasks import OffensiveTask
import re




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }




def config():
    return {
        "dataset": OSACT4SubtaskADataset,
        "task": OffensiveTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["OFF", "NOT_OFF"],
            "max_tries": 10,
        },
    }

def prompt(input_sample):
    return [
        {
            "role": "user",
            "content": (
                f"يرجى تقييم الجملة التالية لتحديد ما إذا كانت تحتوي على محتوى مسيء. إذا كانت الجملة تحتوي على أي درجة من الإساءة، قم بالرد بـ 'مسيء'. إذا لم تحتوي الجملة على أي محتوى مسيء أو مهين أو ضار، قم بالرد بـ 'غير مسيء'.:\n{input_sample}"
            ),
        }
    ]

def post_process(response):
    label = response["choices"][0]["message"]["content"]

    label = label.replace("label:", "").strip().lower()
    label = re.sub(r"<s>|</s>", "", label)

    if "not" in label or "غير" in label or "ليس" in label:
        return "NOT_OFF"
    elif "is" in label or "مسيء" in label:
        return "OFF"
    else:
        return None
