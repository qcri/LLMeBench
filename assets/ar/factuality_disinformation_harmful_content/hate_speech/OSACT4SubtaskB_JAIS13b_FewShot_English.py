from llmebench.datasets import OSACT4SubtaskBDataset
from llmebench.models import FastChatModel
from llmebench.tasks import HateSpeechTask


def metadata():
    return {
        "author": "Mohamed Bayan Kmainasi, Rakif Khan, Ali Ezzat Shahroor, Boushra Bendou, Maram Hasanain, and Firoj Alam",
        "affiliation": "Arabic Language Technologies, Qatar Computing Research Institute (QCRI), Hamad Bin Khalifa University (HBKU)",
        "model": "jais-13b-chat",
        "description": "For a comprehensive analysis and results, refer to our peer-reviewed publication available at [Springer](https://doi.org/10.1007/978-981-96-0576-7_30) or explore the preprint version on [arXiv](https://arxiv.org/abs/2409.07054).",
    }


def config():
    return {
        "dataset": OSACT4SubtaskBDataset,
        "task": HateSpeechTask,
        "model": FastChatModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    out_prompt = base_prompt + "\n"
    for example in examples:
        # Found chatgpt confused when using 0 and 1 in the prompt
        # label = "no" if example["label"] == "0" else "yes"
        label = "not_hate_speech" if example["label"] == "NOT_HS" else "hate_speech"
        out_prompt = (
            out_prompt + "Tweet: " + example["input"] + "\nLabel: " + label + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = out_prompt + "Tweet: " + input_sample + "\nLabel:\n"

    return out_prompt


def prompt(input_sample, examples):
    base_prompt = (
        "Respond only with 'Hate speech' if it is hate speech and 'Not hate speech' if it is not hate speech from the following tweets. "
        "Here are some examples to guide you:\n"
    )
    return [
        {
            "role": "user",
            "content": (few_shot_prompt(input_sample, base_prompt, examples)),
        }
    ]


def post_process(response):
    print(response)
    label = (
        response["choices"][0]["message"]["content"].lower().replace(".", "").strip()
    )
    print("label", label)

    if "NOT" in label or "NO" in label or "لا" in label or "ليس" in label:
        return "NOT_HS"
    return "HS"
