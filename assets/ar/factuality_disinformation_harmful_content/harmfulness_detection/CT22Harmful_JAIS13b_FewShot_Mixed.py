from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HarmfulDetectionTask
import random



def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }



def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": FastChatModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }
def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Translate labels to Arabic
        label = "not_harmful" if example["label"] == "0" else "harmful"
        out_prompt += "التغريدة: " + example["input"] + "\nالتصنيف: " + label + "\n\n"

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt += "التغريدة: " + input_sample + "\nالتصنيف: \n"

    return out_prompt

def prompt(input_sample, examples):
    base_prompt =  "صنف التغريدة التالية على أنها 'harmful' أو 'not_harmful'. قدم التصنيف فقط.\n\n"
    return [
        {
            "role": "user",
            "content": (
                few_shot_prompt(input_sample, base_prompt, examples)
            ),
        }
    ]



def post_process(response):
    label = response["choices"][0]["message"]["content"].lower()
    print("label: ",label)    
    if "لا يمكنني" in label or "ai" in label or "لا يمكن" in label or "ما هي المعايير" in label:
        return None
    if "التغريدة الأخيرة هي 'harmful'" in label:
        return "1"
    if "التغريدة الثالثة مصنفة على أنها 'غير مؤذية'" in label:
        return "0"
    if "التغريدة الثالثة مصنفة على أنها 'مؤذية'" in label or "غير آمن" in label or "لأنها لا تحتوي على محتوى ضار" in label:
        return "1" 
    
    if "label: harmful" in label or "is harmful" in label:
        return "1"
    
    if "غير ضارة" in label or "لا يحتوي على محتوى ضار" in label or "غير" in label or "ليس" in label or "no" in label:
        return "0"
    
    
    else:
        return "1"  
