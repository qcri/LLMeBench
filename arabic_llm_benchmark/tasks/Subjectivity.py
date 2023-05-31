from arabic_llm_benchmark.tasks.task_base import TaskBase

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support

class SubjectivityTask(TaskBase):
	def evaluate(self, gold_labels, pred_labels):
		acc = accuracy_score(gold_labels, pred_labels)
		m_prec, m_rec, m_f1, m_s = precision_recall_fscore_support(gold_labels, pred_labels, average="macro")
		p_prec, p_rec, p_f1, p_s = precision_recall_fscore_support(gold_labels, pred_labels, labels=["SUBJ"])
		results = {
			'accuracy': acc,
			'macro_F1': m_f1,
			'macro_P': m_prec,
			'macro_R': m_rec,
			'SUBJ_F1': p_f1[0],
			'SUBJ_P': p_prec[0],
			'SUBJ_R': p_rec[0],
			# "msg": "performance with respect to the macro average. Ref: CheckThat-2023"
		}

		return results
