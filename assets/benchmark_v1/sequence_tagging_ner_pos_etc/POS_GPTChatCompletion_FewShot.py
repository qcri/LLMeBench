import os
import re
from arabic_llm_benchmark.datasets import ArabicPOSDataset
from arabic_llm_benchmark.models import GPTChatCompletionModel
from arabic_llm_benchmark.tasks import ArabicPOSTask_v4

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
    "NEG_PART":"PART",
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


def config():
    sets = [
        ("egy", "egy.pos/egy.data_5.test.src-trg.sent", "egy.pos/egy.data_5.dev.src-trg.sent"),
        ("glf", "glf.pos/glf.data_5.test.src-trg.sent", "glf.pos/glf.data_5.dev.src-trg.sent"),
        ("mgr", "mgr.pos/mgr.data_5.test.src-trg.sent", "mgr.pos/mgr.data_5.dev.src-trg.sent"),
        ("lev", "lev.pos/lev.data_5.test.src-trg.sent", "lev.pos/lev.data_5.dev.src-trg.sent"),
        ("msa", "WikiNewsTruth.txt.POS.tab", "WikiNewsTruthDev.txt"), 
        ("XGLUE", "XGLUE/ar.test.src-trg.txt", "XGLUE/ar.dev.src-trg.txt")
    ]
    configs = []
    for name, testset, devset in sets:
        configs.append(
            {
                "name": name,
                "config": {
                    "dataset": ArabicPOSDataset,
                    "dataset_args": {},
                    "task": ArabicPOSTask_v4,
                    "task_args": {},
                    "model": GPTChatCompletionModel,
                    "model_args": {
                        "api_type": "azure",
                        "api_version": "2023-03-15-preview",
                        "api_base": os.environ["AZURE_API_URL"],
                        "api_key": os.environ["AZURE_API_KEY"],
                        "engine_name": os.environ["ENGINE_NAME"],
                        # "class_labels": ["m", "f"],
                        "max_tries": 3,
                    },
                    "general_args": {
                        "data_path": "data/sequence_tagging_ner_pos_etc/POS/"
                        + testset, 
                        "fewshot": { 
                            "train_data_path":  "data/sequence_tagging_ner_pos_etc/POS/" + devset
                        }
                    },
                },
            }
        )
    return configs



def few_shot_prompt(input_sample, base_prompt, examples):
    output_prompt = base_prompt + "\n"
    for example in examples:
        tokens = example["input"]
        label = example["label"]
        output_prompt = (
            output_prompt + f"Sentence: {tokens}\nLabels: {label}\n"
        )
    output_prompt = output_prompt + f"Sentence: {input_sample}\n" + "Labels:"
    return output_prompt


def prompt(input_sample, examples): 
    base_prompt = f'Please provide the POS tags for each word in the input sentence.  \
                 The POS tag label should be chosen from the following tag label set: \
                 ["ABBREV", "ADJ", "ADV", "CASE", "CONJ", "DET", "EMOT", "FOREIGN", "FUT_PART", "HASH", "MENTION", "NEG_PART", "NOUN", \
                 "NSUFF", "NUM", "PART", "PREP", "PROG_PART", "PRON", "PUNC", "URL", "V"].'

    return [
            {"role":"system","content": "You are a linguist that helps in annotating data."},
            {
                "role":"user",
                "content": few_shot_prompt(input_sample, base_prompt, examples)
            }
        ]


# def post_process(response):
#     text = response["choices"][0]["message"]["content"]

#     #text = re.sub(r'}[^}]+','}',text)
#     #text = re.sub(r'[^{]+{','{',text)
#     text = re.sub(r"Here's the segmented sentence in a JSON format:",'',text)
#     #print("Pro:",text)
#     pattern = r"[\"\']([^\"\']+)[\'\"]: *[\'\"]([^}]+)[\'\"]"
#     pattern = r"\(\"([^\"]+)\", \"([^\"]+)\"\)"
#     matches = re.finditer(pattern, text)
#     results = []
#     #print("Res0:",results)]
#     if matches: 
#         for m in matches:
#             tag = m.group(2)
#             ntag = []
#             for t in tag.split('+'):
#                 ntag.append(mapTags[t] if t in mapTags else t)
#             results.append('+'.join(ntag))
#     else: 
#         return response["choices"][0]["message"]["content"]

#     #print("Res1:",results)
#     return ' '.join(results)



def post_process(response): 
    return response["choices"][0]["message"]["content"]