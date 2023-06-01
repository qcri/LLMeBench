from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class MachineTranslationTask(TaskBase):
    def __init__(self, **kwargs):
        super(MachineTranslationTask, self).__init__(**kwargs)

    def evaluate(self, reference, candidate):
        score = 0.0
        if(candidate == None):
            candidate = ['']*len(reference)
        if(len(candidate)<len(reference)):
            for i in range(len(reference)-len(candidate)):
                candidate.append([''])

        for i in range(len(reference)):
            score += sentence_bleu(
                [reference[i].strip().split()], candidate[i].strip().split()
            )
        score /= len(reference)
        return {"BLEU Score ": score}
