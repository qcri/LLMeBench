import re
import string
import sys
from collections import Counter

from llmebench.tasks.task_base import TaskBase


class QATask(TaskBase):
    def __init__(self, **kwargs):
        super(QATask, self).__init__(**kwargs)

    # We use the official SQUAD Evaluation script. The following code is adapted from the official squad evaluation script
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def evaluate(self, true_labels, predicted_labels):
        f1, exact_match, total = 0, 0, 0
        for ground_truth, prediction in zip(true_labels, predicted_labels):
            total += 1
            if prediction is None:
                continue
            ground_truths = ground_truth
            exact_match += self.metric_max_over_ground_truths(
                self.exact_match_score, prediction, ground_truths
            )
            f1 += self.metric_max_over_ground_truths(
                self.f1_score, prediction, ground_truths
            )

        # Original script was returning 100* F1 but we report F1 in Larabench so no need to return % here.
        # exact_match = 100.0 * exact_match / total
        # f1 = 100.0 * f1 / total

        exact_match = exact_match / total
        f1 = f1 / total

        return {"exact_match": exact_match, "f1": f1}
