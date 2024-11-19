from llmebench.datasets import CT22HarmfulDataset
from llmebench.models import AzureModel
from llmebench.tasks import HarmfulDetectionTask
import random




def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "Llama-3.1-8B-Instruct",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": CT22HarmfulDataset,
        "task": HarmfulDetectionTask,
        "model": AzureModel,
        "model_args": {
            "class_labels": ["0", "1"],
            "max_tries": 3,
        },
        "general_args": {"test_split": "ar", "fewshot": {"train_split": "ar"}},
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        label = "not_harmful" if example["label"] == "0" else "harmful"
        out_prompt = (
            out_prompt + "tweet: " + example["input"] + "\nlabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt

def prompt(input_sample, examples):
    base_prompt = "Classify the following tweet as 'harmful' or 'not_harmful'. Provide only label."
    return [
        {
            "role": "user",
            "content": (
                few_shot_prompt(input_sample, base_prompt, examples)
            ),
        }
    ]



 
def post_process(response):
        # Extract the label from the response
    if "output" in response:
        label = response["output"].strip().lower()
        
    print("label: ",label)

    
    if "not_harmful" in label:
        return "0"
    elif label==  "harmful":
        return "1"
    
    else:
        return None