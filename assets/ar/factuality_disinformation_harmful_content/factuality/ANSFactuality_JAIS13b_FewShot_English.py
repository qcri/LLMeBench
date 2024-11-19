from llmebench.datasets import ANSFactualityDataset
from llmebench.models import FastChatModel
from llmebench.tasks import FactualityTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": ANSFactualityDataset,
        "task": FactualityTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    prompt_text = "Detect whether the information in the sentence is factually true or false. Answer only by true or false.\n\n"

    fs_prompt = few_shot_prompt(input_sample, prompt_text, examples)
    return [
        {
            "role": "user",
            "content": fs_prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt
    for example in examples:
        sent = example["input"]
        label = example["label"]

        out_prompt = (
            out_prompt + "Sentence: " + sent + "\n" + "label: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Sentence: " + input_sample + "\nlabel: \n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    label = response["choices"][0]["message"]["content"]
    label = label.replace(".", "").strip().lower()


    if (
        "لا" in label
        or "خطأ" in label
        or "ليست" in label
        or "false" in label
        or "label: 0" in label
        or "label: no" in label
        
    ):
        pred_label = "false"
    elif (
        "نعم" in label
        or "صحيحة" in label
        or "true" in label
        or "yes" in label
        or "label: 1" in label
        or "label: yes" in label
    ):
        pred_label = "true"
    else:
        print("label problem!! " + label)
        pred_label = None

    return pred_label
