import regex as re

from llmebench.tasks.task_base import TaskBase


class ArProSpanTask(TaskBase):
    def __init__(self, correct_span=False, **kwargs):
        # Decide whether to correct start and end of gpt predictions
        self.correct_span = correct_span
        super(ArProSpanTask, self).__init__(**kwargs)

    def sort_spans(self, spans):
        """
        sort the list of annotations with respect to the starting offset
        """
        spans = sorted(spans, key=lambda span: span[1][0])

        return spans

    def sort_labels(self, all_labels):
        # print(all_labels)
        sorted_labels = []
        for labels in all_labels:
            per_par_labels = []
            for label in labels:
                start = label["start"]
                end = label["end"]
                if "par_txt" in label:
                    par_txt = label["par_txt"]
                    per_par_labels.append(
                        (label["technique"], [start, end], label["text"], par_txt)
                    )
                else:
                    per_par_labels.append(
                        (label["technique"], [start, end], label["text"])
                    )

            per_par_labels = self.sort_spans(per_par_labels)
            sorted_labels.append(per_par_labels)

        # print(sorted_labels)
        # print(40*"=")

        return sorted_labels

    def reformatLabels(self, true_labels, predicted_labels):
        # filtered_true_labels = []
        # filtered_predicted_labels = []

        # if we apply this, we are like ignoring no technique case at all
        # to match the original scorer from wanlp22 task 2 we have to comment this line out
        # true_labels, predicted_labels = zip(*filter(all, zip(true_labels, predicted_labels)))

        filtered_true_labels = self.sort_labels(list(true_labels))
        filtered_predicted_labels = self.sort_labels(list(predicted_labels))

        return filtered_true_labels, filtered_predicted_labels

    def compute_prec_rec_f1(
        self, prec_numerator, prec_denominator, rec_numerator, rec_denominator
    ):
        p, r, f1 = (0, 0, 0)
        if prec_denominator > 0:
            p = prec_numerator / prec_denominator
        if rec_denominator > 0:
            r = rec_numerator / rec_denominator
        if prec_denominator == 0 and rec_denominator == 0:
            f1 = 1.0
        if p > 0 and r > 0:
            f1 = 2 * (p * r / (p + r))

        return p, r, f1

    def span_intersection(self, gold_span, pred_span):
        x = range(gold_span[0], gold_span[1])
        y = range(pred_span[0], pred_span[1])
        inter = set(x).intersection(y)
        return len(inter)

    def compute_technique_frequency(self, annotations, technique_name):
        all_annotations = []
        for annot in annotations:
            for x in annot:
                all_annotations.append(x[0])

        techn_freq = sum([1 for a in all_annotations if a == technique_name])

        # print(technique_name,techn_freq)

        return techn_freq

    def ammend_span(self, span, span_txt, par):
        start = span[0]
        end = span[1]

        try:
            # get the first matching span
            for match in re.finditer(span_txt, par):
                start = match.start()
                end = match.end()
                break
        except:
            print("Error start end correction")

        return [start, end]

    def compute_span_score(self, gold_annots, pred_annots):
        # count total no of annotations
        rec_denominator = sum([len(x) for x in gold_annots])
        prec_denominator = sum([len(x) for x in pred_annots])

        techniques = self.dataset.get_predefined_techniques()
        techniques.remove("no_technique")

        technique_Spr_prec = {
            propaganda_technique: 0 for propaganda_technique in techniques
        }
        technique_Spr_rec = {
            propaganda_technique: 0 for propaganda_technique in techniques
        }
        cumulative_Spr_prec, cumulative_Spr_rec = (0, 0)
        f1_articles = []

        for i, pred_annot_obj in enumerate(pred_annots):
            gold_annot_obj = gold_annots[i]
            # print("%s\t%d\t%d" % (example_id, len(gold_annot_obj), len(pred_annot_obj)))

            document_cumulative_Spr_prec, document_cumulative_Spr_rec = (0, 0)
            for j, pred_ann in enumerate(pred_annot_obj):
                s = ""
                ann_length = pred_ann[1][1] - pred_ann[1][0]

                for i, gold_ann in enumerate(gold_annot_obj):
                    if pred_ann[0] == gold_ann[0]:
                        if self.correct_span:
                            pred_ann = list(pred_ann)
                            # We get the paragraph from the gold par (gold_ann[3]) and the
                            # predicted span text from pred_ann[2]
                            pred_ann[1] = self.ammend_span(
                                pred_ann[1], pred_ann[2], gold_ann[3]
                            )
                            pred_ann = tuple(pred_ann)

                        # s += "\tmatch %s %s-%s - %s %s-%s"%(sd[0],sd[1], sd[2], gd[0], gd[1], gd[2])
                        intersection = self.span_intersection(gold_ann[1], pred_ann[1])
                        # print(intersection)
                        # print(intersection)
                        s_ann_length = gold_ann[1][1] - gold_ann[1][0]
                        Spr_prec = intersection / ann_length
                        document_cumulative_Spr_prec += Spr_prec
                        cumulative_Spr_prec += Spr_prec
                        s += (
                            "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|p| = %d/%d = %f (cumulative S(p,r)=%f)\n"
                            % (
                                pred_ann[0],
                                pred_ann[1][0],
                                pred_ann[1][1],
                                gold_ann[0],
                                gold_ann[1][0],
                                gold_ann[1][1],
                                intersection,
                                ann_length,
                                Spr_prec,
                                cumulative_Spr_prec,
                            )
                        )
                        technique_Spr_prec[gold_ann[0]] += Spr_prec

                        Spr_rec = intersection / s_ann_length
                        document_cumulative_Spr_rec += Spr_rec
                        cumulative_Spr_rec += Spr_rec
                        s += (
                            "\tmatch %s %s-%s - %s %s-%s: S(p,r)=|intersect(r, p)|/|r| = %d/%d = %f (cumulative S(p,r)=%f)\n"
                            % (
                                pred_ann[0],
                                pred_ann[1][0],
                                pred_ann[1][1],
                                gold_ann[0],
                                gold_ann[1][0],
                                gold_ann[1][1],
                                intersection,
                                s_ann_length,
                                Spr_rec,
                                cumulative_Spr_rec,
                            )
                        )
                        technique_Spr_rec[gold_ann[0]] += Spr_rec

            p_article, r_article, f1_article = self.compute_prec_rec_f1(
                document_cumulative_Spr_prec,
                len(pred_annot_obj),
                document_cumulative_Spr_rec,
                len(gold_annot_obj),
            )
            f1_articles.append(f1_article)

        p, r, f1 = self.compute_prec_rec_f1(
            cumulative_Spr_prec, prec_denominator, cumulative_Spr_rec, rec_denominator
        )

        f1_per_technique = []

        for technique_name in technique_Spr_prec.keys():
            prec_tech, rec_tech, f1_tech = self.compute_prec_rec_f1(
                technique_Spr_prec[technique_name],
                self.compute_technique_frequency(pred_annots, technique_name),
                technique_Spr_prec[technique_name],
                self.compute_technique_frequency(gold_annots, technique_name),
            )
            f1_per_technique.append(f1_tech)

        return p, r, f1, f1_per_technique

    def evaluate(self, true_labels, predicted_labels):
        # fix none labels by empty lists instead of randomized predictions
        predicted_labels = [p if p else [] for p in predicted_labels]

        # gold_labels_set = set(itertools.chain.from_iterable(true_labels))

        true_labels, predicted_labels = self.reformatLabels(
            true_labels, predicted_labels
        )

        # for p in predicted_labels:
        # if p == None or len(p) == 0:
        # p = [self.get_random_prediction(gold_labels_set) for _ in range(len(t))]

        precision, recall, micro_f1, f1_per_class = self.compute_span_score(
            true_labels, predicted_labels
        )
        macro_f1 = sum(f1_per_class) / len(f1_per_class)

        return {
            "Micro F1": micro_f1,
            "Macro F1": macro_f1,
            "Precision": precision,
            "Recall": recall,
        }
