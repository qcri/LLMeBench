import re

from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class ArabicPOSTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicPOSTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # split sentence into words
        attrs = vars(self)
        hyp = []
        ref = []

        for t, p in zip(true_labels, predicted_labels):

            #print("tt:",t)
            #print("pp:",p)
            if p == None:
                continue
            if p is None or ("Sorry, I cannot") in p:
                # print("Sorry!")
                p = None
            #print("P:",type(p),len(p), p)
            if p == None:
                # return unsegmented text!
                p = [""] * len(t)

            t = t.replace('+NSUFF','').replace('DET+','')
            p = p.replace('+NSUFF','').replace('DET+','')

            p = p.split()
            t = t.split()
            if len(p) < len(t):
                for i in range(len(t) - len(p)):
                    p.append("")

            hyp += p[: len(t)]
            ref += t
        #print("ph:",len(hyp),len(ref))
        return {"Macro F1": f1_score(ref, hyp, average="macro")}
