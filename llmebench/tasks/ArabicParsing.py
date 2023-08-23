import re

from sklearn.metrics import f1_score

from llmebench.tasks.task_base import TaskBase


class ArabicParsingTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicParsingTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        hyp = []
        ref = []
        for tdict, pdict in zip(true_labels, predicted_labels):
            thyp = {}
            if pdict == None:
                for i in tdict:
                    thyp[i] = "0"
            else:
                for p in pdict:
                    thyp[p] = pdict[p]

            for l in tdict:
                ref.append(tdict[l])
                if l in thyp:
                    hyp.append(thyp[l])
                else:
                    hyp.append(0)

        return {"Macro F1": f1_score(ref, hyp, average="macro")}
