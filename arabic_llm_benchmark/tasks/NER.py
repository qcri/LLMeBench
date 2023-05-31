from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class NERTask(TaskBase):
    def __init__(self, **kwargs):
        super(NERTask, self).__init__(**kwargs)

    def evaluate(self, ref_list, pred_list):
        average_score = 0
        scores = [] 
        to_inspect = []
        for i, elem in enumerate(pred_list):
            if len(ref_list[i]) == len(elem):
                f1 = f1_score(ref_list[i], elem, average= "macro")
                scores.append(f1)
            elif len(elem) < len(ref_list[i]):
                while len(elem) < len(ref_list[i]):
                    elem += ["O"]
                f1 = f1_score(ref_list[i], elem, average= "macro")
                scores.append(f1)
            else:
                elem = elem[:len(ref_list[i])]
                f1 = f1_score(ref_list[i], elem, average= "macro")
                scores.append(f1)
            average_score += f1
        average_score /= len(pred_list)
        return {
            "Macro F1 per sample": scores,
            "Macro F1": average_score
        }
