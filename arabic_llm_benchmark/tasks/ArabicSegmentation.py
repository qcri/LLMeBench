import re

from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class ArabicSegmentationTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicSegmentationTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # split sentence into words
        attrs = vars(self)
        # print("P:",len(predicted_labels), " t:",len(true_labels))
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
                s = (
                    re.sub("\([^\)]+\)", "", p)
                    .replace("+}", "}")
                    .replace("}+", "+")
                    .replace("+{", "+")
                    .replace(": {", ": ")
                    .replace("}}", "}")
                )
                s = re.sub(r":\s?(?![{\[\s])([^,}]+)", r': "\1"', s)
                s = re.sub(r"{([^:]+):", r'{"\1":', s)
                s = list(eval(s))
                p = " ".join(["".join([e[v] for v in e]) for e in s])
            else:
                p = None
            # remove punctuation!
            t = re.sub(r"[^\w+\+]", " ", t)
            t = t.split()
            if p == None:
                p = [""] * len(t)
            else:
                p = p.split()

            if len(p) < len(t):
                for i in range(len(t) - len(p)):
                    p.append("")
            # print("PP1:",len(p),p)
            # print("TT1:",len(t),t)
            hyp += p[: len(t)]
            ref += t
        # print("ph:",len(hyp),hyp)
        # print("tt:",len(ref),ref)
        return {"Macro F1": f1_score(ref, hyp, average="macro")}
