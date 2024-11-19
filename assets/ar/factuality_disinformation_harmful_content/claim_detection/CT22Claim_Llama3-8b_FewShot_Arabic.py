from llmebench.datasets import CT22ClaimDataset
from llmebench.models import AzureModel
from llmebench.tasks import ClaimDetectionTask




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }




def config():
    return {
        "dataset": CT22ClaimDataset,
        "task": ClaimDetectionTask,
        "model": AzureModel,
        "model_args": { 
            "max_tries": 30,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }

def prompt(input_sample, few_shot_examples):
    few_shot_text = ""
    for example in few_shot_examples:
        few_shot_text += (
            "الجملة: " + example["input"] + "\n"
            + "الإجابة: " + ("نعم" if example["label"] == "1" else "لا") + "\n\n"
        )

    return [
        {
            "role": "user",
            "content": (
                "هل تحتوي هذه الجملة على ادعاء يمكن التحقق منه؟ أجب فقط بـ 'نعم' أو 'لا'. قدم فقط الإجابة.\n\n"
                + few_shot_text
                + "الجملة: " + input_sample + "\n"
                + "الإجابة: "
            )
        }
    ]
import random
def post_process(response):
    try:
        label = ""

        if "output" in response:
            label = response["output"].strip().lower()

        print(f"Extracted Label: {label}")
        if "لا أستطيع" in label or "I cannot" in label:
            return random.choice(["0","1"])

        if "نعم" in label:
            pred_label = "1"
        elif "لا" in label:
            pred_label = "0"
        else:
            pred_label = ""

        print(f"Predicted Label: {pred_label}")

        return pred_label
    except Exception as e:
        print(f"Error in post-processing: {str(e)}")
        return "0"
