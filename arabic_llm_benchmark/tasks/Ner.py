from sklearn.metrics import f1_score

from arabic_llm_benchmark.tasks.task_base import TaskBase


class Ner(TaskBase):
    def __init__(self, **kwargs):
        super(Ner, self).__init__(**kwargs)

    

    def evaluate(self, true_labels, predicted_labels):
        all_ground_truths = [] 
        all_predictions = []
        failed_count = 0
        # define an inner function to clean the ground truth. 
        def clean_ground_truth(ground_truth): 
            cleaned_version = []
            for i, elem in enumerate(ground_truth): 
                if "I-MIS" in elem: 
                    cleaned_version.append("I-MISC") 
                elif "B-MIS" in elem: 
                    cleaned_version.append("B-MISC") 
                else: 
                    cleaned_version.append(elem)
            return cleaned_version
        
        for i, elem in enumerate(true_labels): 
            pred_labels = predicted_labels[i] 
            if pred_labels is None: 
                failed_count += 1 
                continue 
            ground_truth = clean_ground_truth(elem.split())

            if len(pred_labels) == 0: 
                pred_labels = ["O"] * len(ground_truth) 

            if len(ground_truth) == len(pred_labels): 
                all_ground_truths.extend(ground_truth) 
                all_predictions.extend(pred_labels) 
            
            elif len(pred_labels) < len(ground_truth): 
                while len(pred_labels) < len(ground_truth): 
                    pred_labels += ["O"] 
                all_ground_truths.extend(ground_truth) 
                all_predictions.extend(pred_labels)
            else: 
                pred_labels = pred_labels[:len(ground_truth)]
                all_ground_truths.extend(ground_truth)
                all_predictions.extend(pred_labels)


        return {"Macro F1": f1_score(all_ground_truths, all_predictions, average="macro")}
