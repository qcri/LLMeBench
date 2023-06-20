import re

from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class ArabicSegmentationTask_v4(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicSegmentationTask_v4, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # split sentence into words
        attrs = vars(self)
        # print("P:",len(predicted_labels), " t:",len(true_labels))
        hyp = []
        ref = []
        for t, p in zip(true_labels, predicted_labels):
            # print("P0:", p)
            if p == None:
                # return unsegmented text!
                p = t.replace("+", "").split()
            else:
                p = p.split()

            t = re.sub(r"[^ ]+[A-Za-z]+ ", " ", t)

            t = t.split()

            if len(p) < len(t):
                for i in range(len(t) - len(p)):
                    p.append("")

            # if(len(p)!=len(t)):
            print("PP1:", len(p[: len(t)]), p[: len(t)])
            print("TT1:", len(t), t)

            hyp += p[: len(t)]
            ref += t
        # print("ph:",len(hyp),hyp)
        # print("tt:",len(ref),ref)
        return {"Macro F1": f1_score(ref, hyp, average="macro")}
