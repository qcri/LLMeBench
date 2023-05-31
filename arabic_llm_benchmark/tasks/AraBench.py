from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class AraBenchTask(TaskBase):
    def __init__(self, **kwargs):
        super(AraBenchTask, self).__init__(**kwargs)

    def evaluate(self, reference, candidate):
        score = 0.0
        for i in range(len(reference)):
            score += sentence_bleu(
                [reference[i].strip().split()], candidate[i].strip().split()
            )
        score /= len(reference)
        return {"BLEU Score ": score}
