import re

from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


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
    "": "",
}


class ArabicPOSTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicPOSTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # split sentence into words
        attrs = vars(self)
        hyp = []
        ref = []
        for t, p in zip(true_labels, predicted_labels):
            # print("P:",type(p),len(p), p)
            if p is None or ("Sorry, I cannot") in p:
                # print("Sorry!")
                p = None
            elif "'+ '" in p:
                # Result as raw text
                p = re.sub(r"'\+ '", "+", str(p))
                s = list(eval(p))
                p = " ".join(["".join([e[v] for v in e]) for e in s])
            elif ": " in p:
                # Result as pseudo json
                pattern = r"(\w+)(\W)*:(\W)*([^\']+)'"
                matches = re.finditer(pattern, p)
                p = " ".join([m.group(4) for m in matches])
            else:
                p = None
            # # remove punctuation!
            # t = re.sub(r"[^\w+\+]", " ", t)
            if p == None:
                # return unsegmented text!
                p = [""] * len(t)
            else:
                p = p.split()

            t = t.split()

            if len(p) < len(t):
                for i in range(len(t) - len(p)):
                    p.append("")

            mp = []
            for tag in p:
                tag = tag.replace("'", "")
                ntag = []
                for e in tag.split("+"):
                    if e in mapTags:
                        ntag.append(mapTags[e])
                    else:
                        ntag.append("")
                mp.append("+".join(ntag))
            p = mp

            # p = [mapTags[tag.replace('\'','')] for tag in p]
            # print("PP1:",len(p),p)
            # print("TT1:",len(t),t)
            hyp += p[: len(t)]
            ref += t
        # print("ph:",len(hyp),hyp)
        # print("tt:",len(ref),ref)
        return {"Macro F1": f1_score(ref, hyp, average="macro")}
