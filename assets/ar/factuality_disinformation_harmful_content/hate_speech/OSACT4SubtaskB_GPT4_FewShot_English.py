from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import HateSpeechTask





def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "GPT-4o-2024-05-22",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054)."
    }





def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample, examples):
    base_prompt = (
        "Does the tweet contain hate speech? Answer by "
        " 'hate speech' if it does, and 'not hate speech' otherwise. Here are some examples:\n"
    )

    return [
        {
            "role": "system",
            "content": "You are an expert in tweets analysis and classification.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        # label = "no" if example["label"] == "0" else "yes"
        label = "not hate speech" if example["label"] == "NOT_HS" else "hate speech"
        out_prompt = (
            out_prompt + "tweet: " + example["input"] + "\nlabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "tweet: " + input_sample + "\nlabel:\n"

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    out = response["choices"][0]["message"]["content"]
    label = out.lower().strip()

    if "ليس" in label or "ليس كراهية" in label or "لا" in label or "no" in label or "not" in label or "don't" in label or "not_hs" in label or "not_hatespeech" in label or "not_hate_speech" in label:
        return "NOT_HS"
    elif "كراهية" in label or "نعم" in label or "أجل" in label or "yes" in label or "contins" in label or "hs" in label or "hatespeech" in label or "hate speech" in label:
        return "HS"
    else:
        return None
