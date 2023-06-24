import re

from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class ArabicPOSTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicPOSTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        hyp = []
        ref = []

        for t, p in zip(true_labels, predicted_labels):
            t = t.split()
            if p == None:
                p = [""] * len(t)
            else:
                p = p.split()

                # If prediction is missing tokens, pad with empty tokens
                if len(p) < len(t):
                    for i in range(len(t) - len(p)):
                        p.append("")

                # If prediction has extra tokens, only consider the first
                # N tokens, where N == number of gold tokens
                p = p[: len(t)]

            hyp += p
            ref += t

        return {"Macro F1": f1_score(ref, hyp, average="macro")}
