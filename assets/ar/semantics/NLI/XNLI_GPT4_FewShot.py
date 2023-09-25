from llmebench.datasets import XNLIDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import XNLITask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Accuracy": "0.774"},
    }


def config():
    return {
        "dataset": XNLIDataset,
        "task": XNLITask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
        "general_args": {"fewshot": {"train_split": "dev"}},
    }


def prompt(input_sample, examples):
    prompt_text = "You are provided with a premise and a hypothesis. Your task is to classify the hypothesis as either true (entailing the premise), false (contradicting the premise), or unknown (neutral) based on the given premise. The output should only be exactly one of three labels: true, false or unknown."

    fs_prompt = few_shot_prompt(input_sample, prompt_text, examples)
    return [
        {
            "role": "system",
            "content": "You are an expert in Arabic language understanding.",
        },
        {
            "role": "user",
            "content": fs_prompt,
        },
    ]


def few_shot_prompt(input_sample, base_prompt, examples):
    sent1, sent2 = input_sample.split("\t")

    out_prompt = base_prompt + "\n"
    for example in examples:
        ex_sent1, ex_sent2 = example["input"].split("\t")
        label = example["label"]

        out_prompt = (
            out_prompt
            + "Premise: "
            + ex_sent1
            + "\nHypothesis: "
            + ex_sent2
            + "\n"
            + "label: "
            + label
            + "\n\n"
        )

    # Append the sentence we want the model to predict for but leave the Label blank
    out_prompt = (
        out_prompt + "Premise: " + sent1 + "\nHypothesis: " + sent2 + "\nlabel: \n"
    )

    # print("=========== FS Prompt =============\n")
    # print(out_prompt)

    return out_prompt


def post_process(response):
    input_label = response["choices"][0]["message"]["content"]
    input_label = input_label.replace(".", "").strip().lower()

    if "neutral" in input_label or "unknown" in input_label:
        pred_label = "neutral"
    elif "true" in input_label or "entailment" in input_label:
        pred_label = "entailment"
    elif "false" in input_label or "contradiction" in input_label:
        pred_label = "contradiction"
    else:
        print(input_label)
        pred_label = None

    return pred_label
