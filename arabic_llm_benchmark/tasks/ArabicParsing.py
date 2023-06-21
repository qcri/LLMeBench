import re

from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase



class ArabicParsingTask(TaskBase):
    def __init__(self, **kwargs):
        super(ArabicParsingTask, self).__init__(**kwargs)

    def evaluate(self, true_labels, predicted_labels):
        # FLATTEN both arrays
        #
        #     true_labels:
        #         {0: 0
        #         1: 2}
        # # predicted_labels
        #     0\t1
        #     1\t2
        hyp = []
        ref = []
        for tdict, pdict in zip(true_labels, predicted_labels):
            thyp = {}
            #print("p:",pdict)
            #print("t:",tdict)
            if pdict is None:
                for i in tdict:
                    thyp.append(0)
            else:
                for p in pdict:
                    thyp[p] = pdict[p]

            for l in tdict:
                ref.append(tdict[l])
                if(l in thyp):
                    hyp.append(thyp[l])
                else:
                    hyp.append(0)

        #print("H:",hyp)
        #print("R:",ref)
        return {"Macro F1": f1_score(ref, hyp, average="macro")}
