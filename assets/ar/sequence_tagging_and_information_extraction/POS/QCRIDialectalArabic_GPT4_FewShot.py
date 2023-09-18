import re

from llmebench.datasets import QCRIDialectalArabicPOSDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ArabicPOSTask

mapTags = {
    "UNK": "UNK",
    "EOS": "EOS",
    ".": "PUNC",
    "NNP": "NOUN",
    "JJR": "ADJ",
    "CD": "NOUN",
    "ADJ": "ADJ",
    "adjective": "ADJ",
    "JJ": "ADJ",
    "MD": "ADJ",
    "ADJF": "ADJ",
    "ADV": "ADV",
    "ADP": "ADV",
    "adverb": "ADV",
    "conjunction": "CONJ",
    "CONJ": "CONJ",
    "SCONJ": "CONJ",
    "CCONJ": "CONJ",
    "CC": "CONJ",
    "DT": "DET",
    "DET": "DET",
    "hashtag": "HASH",
    "NOUN": "NOUN",
    "noun": "NOUN",
    "N": "NOUN",
    "NN": "NOUN",
    "INTJ": "NOUN",
    "PROPN": "NOUN",
    "NEG": "PART",
    "PART": "PART",
    "NEG_PART": "PART",
    "IN": "PART",
    "preposition": "PREP",
    "P": "PREP",
    "PREP": "PREP",
    "PRP": "PREP",
    "PRON": "PRON",
    "pronoun": "PRON",
    "REL": "PRON",
    "DEM": "PRON",
    "PUNC": "PUNC",
    "punctuation": "PUNC",
    "PUNCT": "PUNC",
    "SYM": "PUNC",
    "verb": "V",
    "VERB": "V",
    "V": "V",
    "VB": "V",
    "RB": "ADV",
    "VBG": "V",
    "VBZ": "V",
    "PRO": "PRON",
    "conj": "CONJ",
    "punct": "PUNC",
    "neg": "PART",
    "pron": "PRON",
    "prep": "PREP",
    "COMP": "ADJ",
    "interjection": "PART",
    "number": "NOUN",
    "MOD": "PART",
    "NUM": "NOUN",
    "determiner": "DET",
    "negation": "PART",
    "url": "URL",
    "demonstrative": "PRON",
    "particle": "PART",
    "HASHTAG": "HASH",
    "NPROP": "NOUN",
    "EMOJI": "EMOJI",
    ",": "PUNC",
    "RELPRO": "PRON",
    "X": "NOUN",
    "MENTION": "MENTION",
    "اسم": "NOUN",
    "اسم علم": "NOUN",
    "حرف جر": "PREP",
    "حرف شرطي": "PART",
    "حرف عطف": "CONJ",
    "حرف نداء": "PART",
    "حرف نفي": "PART",
    "عدد": "NOUN",
    "فاصلة": "ADJ",
    "فعل": "V",
    "": "",
}


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4-32k (version 0314)",
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'. 3 samples where chosen per test sample based on MaxMarginalRelevance for few shot learning.",
        "scores": {"Accuracy": "0.323"},
    }


def config():
    return {
        "dataset": QCRIDialectalArabicPOSDataset,
        "task": ArabicPOSTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 30,
        },
        "general_args": {
            "fewshot": {
                "train_split": [
                    "glf.data_5/dev",
                    "lev.data_5/dev",
                    "egy.data_5/dev",
                    "mgr.data_5/dev",
                ],
            }
        },
    }


def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"]
        label = example["label"]
        sample = list(zip(tokens.split(), label.split()))
        output_prompt = (
            output_prompt
            + f"Sentence: {tokens.split()}"
            + "\n"
            + f"Labels: {sample}"
            + "\n"
        )
    output_prompt = (
        output_prompt + f"Sentence: {input_sample.split()}" + "\n" + "Labels:"
    )
    return output_prompt


def prompt(input_sample, examples):
    base_prompt = f'Please provide the POS tags for each word in the input sentence. The input will be a list of words in the sentence. The output format should be a list of tuples, where each tuple consists of a word from the input text and its corresponding POS tag label from the tag label set: ["ABBREV", "ADJ", "ADV", "CASE", "CONJ", "DET", "EMOT", "FOREIGN", "FUT_PART", "HASH", "MENTION", "NEG_PART", "NOUN", "NSUFF", "NUM", "PART", "PREP", "PROG_PART", "PRON", "PUNC", "URL", "V"]. Note: Your response should include only a list of tuples, in the order that the words appear in the input sentence, with each tuple containing the corresponding POS tag label for a word.'

    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content": few_shot_prompt(input_sample, base_prompt, examples),
        },
    ]


def post_process(response):
    text = response["choices"][0]["message"]["content"]
    matches = re.findall(r"\((.*?)\)", text)
    if matches:
        cleaned_response = []
        for match in matches:
            elements = match.split(",")
            try:
                cleaned_response.append(elements[1])
            except:
                if ":" in elements[0]:
                    cleaned_response.append("EMOT")
                elif len(elements[0].replace("'", "").strip()) == 0:
                    cleaned_response.append("PUNCT")

        cleaned_response = [
            sample.replace("'", "").strip() for sample in cleaned_response
        ]
        cleaned_response = " ".join(cleaned_response)
    else:
        cleaned_response = None
    return cleaned_response
