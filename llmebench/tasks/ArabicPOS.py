import itertools

from sklearn.metrics import accuracy_score

from llmebench.tasks.task_base import TaskBase


class ArabicPOSTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicPOSTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        hyp = []
        ref = []

        gold_labels_set = set(itertools.chain.from_iterable(true_labels))

        for t, p in zip(true_labels, predicted_labels):
            t = t.split()
            if p == None:
                p = [self.get_random_prediction(gold_labels_set) for _ in range(len(t))]
            else:
                p = p.split()

                # If prediction is missing tokens, pad with empty tokens
                if len(p) < len(t):
                    for i in range(len(t) - len(p)):
                        p.append(self.get_random_prediction(gold_labels_set))

                # If prediction has extra tokens, only consider the first
                # N tokens, where N == number of gold tokens
                p = p[: len(t)]

            hyp += p
            ref += t

        return {"Accuracy": accuracy_score(ref, hyp)}
