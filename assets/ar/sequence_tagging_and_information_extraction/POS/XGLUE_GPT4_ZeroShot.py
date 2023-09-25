import re

from llmebench.datasets import XGLUEPOSDataset
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
        "description": "GPT4 32k tokens model hosted on Azure, using the ChatCompletion API. API version '2023-03-15-preview'.",
        "scores": {"Accuracy": "0.402"},
    }


def config():
    return {
        "dataset": XGLUEPOSDataset,
        "task": ArabicPOSTask,
        "model": OpenAIModel,
        "model_args": {
            "max_tries": 3,
        },
    }


def prompt(input_sample):
    return [
        {
            "role": "system",
            "content": "You are a linguist that helps in annotating data.",
        },
        {
            "role": "user",
            "content":
            # f"Assign POS tag to each morphological segment within each word. group the tags for each word with +: {input_sample}"
            # + ".\nThe output should be in the format: [{word: label}, {word: label}]",
            f'Please provide the POS tags for each word in the input sentence. The input will be a list of words in the sentence. \
                 The output format should be a list of tuples, where each tuple consists of a word from the input text and its \
                 corresponding POS tag label from the tag label set: \
                 ["ABBREV", "ADJ", "ADV", "CASE", "CONJ", "DET", "EMOT", "FOREIGN", "FUT_PART", "HASH", "MENTION", "NEG_PART", "NOUN", \
                 "NSUFF", "NUM", "PART", "PREP", "PROG_PART", "PRON", "PUNC", "URL", "V"].\
                Note: Your response should include only a list of tuples, in the order that the words appear in the input sentence, \
                with each tuple containing the corresponding POS tag label for a word. Input:+: {input_sample}',
        },
    ]


def post_process(response):
    text = response["choices"][0]["message"]["content"]

    if "Sorry, I cannot" in text or "Unfortunately" in text:
        return None

    text = re.sub(r"Here's the segmented sentence in a JSON format:", "", text)

    pattern = r"\(\"([^\"]+)\", \"([^\"]+)\"\)"
    matches = re.finditer(pattern, text)
    results = []

    for m in matches:
        tag = m.group(2)
        ntag = []
        for t in tag.split("+"):
            ntag.append(mapTags[t] if t in mapTags else t)
        results.append("+".join(ntag))

    return " ".join(results)
