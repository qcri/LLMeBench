import evaluate
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

from llmebench.tasks.task_base import TaskBase


class MultiNativQATask(TaskBase):
    def __init__(self, **kwargs):
        super(MultiNativQATask, self).__init__(**kwargs)

    def evaluate(self, references, candidates):
        nltk.download("wordnet")
        candidates = [c if c else "" for c in candidates]
        bleu_score = corpus_bleu([[r] for r in references], candidates)

        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(predictions=candidates, references=references)

        def corpus_meteor(predicted, references):
            meteor_score_sentences_list = list()
            for reference, predict in zip(references, predicted):
                meteor_score_sentences_list.append(
                    meteor_score([word_tokenize(reference)], word_tokenize(predict))
                )
            meteor_score_res = np.mean(meteor_score_sentences_list)
            return meteor_score_res

        meteor_scores = corpus_meteor(candidates, references)

        return {"BLEU": bleu_score, "ROUGE": rouge_score, "METEOR": meteor_scores}
