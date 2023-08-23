import re

from sklearn.metrics import accuracy_score

from llmebench.tasks.task_base import TaskBase


class ArabicSegmentationTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicSegmentationTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        hyp = []
        ref = []
        for t, p in zip(true_labels, predicted_labels):
            if p == None:
                # Use unsegmented text on prediction failure
                p = t.replace("+", "").split()
            else:
                p = p.split()

            t = t.split()

            # If prediction is missing tokens, pad with empty tokens
            if len(p) < len(t):
                for i in range(len(t) - len(p)):
                    p.append("")

            # If prediction has extra tokens, only consider the first
            # N tokens, where N == number of gold tokens
            hyp += p[: len(t)]
            ref += t

        return {"Accuracy": accuracy_score(ref, hyp)}
