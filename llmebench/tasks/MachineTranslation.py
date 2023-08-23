from nltk.translate.bleu_score import corpus_bleu

from llmebench.tasks.task_base import TaskBase


class MachineTranslationTask(TaskBase):
    def __init__(self, **kwargs):
        super(MachineTranslationTask, self).__init__(**kwargs)

    def evaluate(self, references, candidates):
        candidates = [c if c else "" for c in candidates]
        bleu_score = corpus_bleu([[r] for r in references], candidates)

        return {"BLEU": bleu_score}
