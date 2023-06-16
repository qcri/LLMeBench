from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class AnerCorpDataset(DatasetBase):
    def __init__(self, test_filenames, **kwargs):
        super(AnerCorpDataset, self).__init__(**kwargs)

    def citation(self):
        return """ https://camel.abudhabi.nyu.edu/anercorp/ """

    def get_data_sample(self):
        return {
            "input": ".كانت السبب الرئيس في سقوط البيزنطيين بسبب الدمار الذي كانت تخلفه الحملات الأولى المارة في بيزنطة ( مدينة القسطنطينية ) عاصمة الإمبراطورية البيزنطية وتحول حملات لاحقة نحوها",
            "label": "O O O O O B-PER O O O O O O O O O B-LOC O O B-LOC O O B-LOC I-LOC O O O O O",
        }

    def load_data(self, data_path, no_labels=False):
        data = []
        with open(data_path, "r") as reader:
            current_sentence = []
            current_label = []
            for line_idx, line in enumerate(reader):
                if len(line.strip()) == 0:
                    sentence = " ".join(current_sentence)
                    label = " ".join(current_label)
                    data.append(
                        {"input": sentence, "label": label, "line_number": line_idx}
                    )
                else:
                    elements = line.strip().split()
                    current_sentence.append(elements[0])
                    current_label.append(elements[1])
        return data
