from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from llmebench.tasks.task_base import TaskBase


class SubjectivityTask(TaskBase):
    def __init__(self, **kwargs):
        super(SubjectivityTask, self).__init__(**kwargs)

    def evaluate(self, gold_labels, pred_labels):
        pred_labels = [
            p if p else self.get_random_prediction(set(gold_labels))
            for p in pred_labels
        ]
        acc = accuracy_score(gold_labels, pred_labels)
        m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(
            gold_labels, pred_labels, average="macro"
        )
        p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(
            gold_labels, pred_labels, labels=["SUBJ"]
        )
        # Calculate weighted precision, recall, and F1 score
        precision = precision_score(
            gold_labels, pred_labels, average="weighted", labels=["SUBJ"]
        )
        recall = recall_score(
            gold_labels, pred_labels, average="weighted", labels=["SUBJ"]
        )
        f1 = f1_score(gold_labels, pred_labels, average="weighted", labels=["SUBJ"])
        results = {
            "accuracy": acc,
            "macro_F1": m_f1,
            "macro_P": m_prec,
            "macro_R": m_rec,
            "SUBJ_F1": p_f1[0],
            "SUBJ_P": p_prec[0],
            "SUBJ_R": p_rec[0],
            "W_SUBJ_F1": precision,
            "W_SUBJ_P": recall,
            "W_SUBJ_R": f1,
            # "msg": "performance with respect to the macro average. Ref: CheckThat-2023"
        }

        return results
