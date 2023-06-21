from pathlib import Path

from arabic_llm_benchmark.datasets.dataset_base import DatasetBase


class AqmarDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(AqmarDataset, self).__init__(**kwargs)
        self.dev_filenames = kwargs.get("dev_filenames", [
                    "Damascus.txt", 
                    "Atom.txt",
                    "Raul_Gonzales.txt", 
                    "Linux.txt", 
                    "Imam_Hussein_Shrine.txt",
                    "Nuclear_Power.txt", 
                    "Real_Madrid.txt", 
                    "Solaris.txt"

            ])
        self.test_filenames = kwargs.get("test_filenames", [
                "Crusades.txt",
                "Islamic_Golden_Age.txt",
                "Islamic_History.txt",
                "Ibn_Tolun_Mosque.txt",
                "Ummaya_Mosque.txt",
                "Enrico_Fermi.txt",
                "Light.txt",
                "Periodic_Table.txt",
                "Physics.txt",
                "Razi.txt",
                "Summer_Olympics2004.txt",
                "Christiano_Ronaldo.txt",
                "Football.txt",
                "Portugal_football_team.txt",
                "Soccer_Worldcup.txt",
                "Computer.txt",
                "Computer_Software.txt",
                "Internet.txt",
                "Richard_Stallman.txt",
                "X_window_system.txt",
            ])

    def citation(self):
        return """@inproceedings{mohit-etal-2012-recall,
                title = \"Recall-Oriented Learning of Named Entities in {A}rabic {W}ikipedia\",
                author = \"Mohit, Behrang  and
                Schneider, Nathan  and
                Bhowmick, Rishav  and
                Oflazer, Kemal  and
                Smith, Noah A.\",
                booktitle = \"Proceedings of the 13th Conference of the {E}uropean Chapter of the Association for Computational Linguistics\",
                month = apr,
                year = \"2012\",
                address = \"Avignon, France\",
                publisher = \"Association for Computational Linguistics\",
                url = \"https://aclanthology.org/E12-1017\",
                pages = \"162--173\",
}
        }"""

    def get_data_sample(self):
        return {
            "input": ".كانت السبب الرئيس في سقوط البيزنطيين بسبب الدمار الذي كانت تخلفه الحملات الأولى المارة في بيزنطة ( مدينة القسطنطينية ) عاصمة الإمبراطورية البيزنطية وتحول حملات لاحقة نحوها",
            "label": "O O O O O B-PER O O O O O O O O O B-LOC O O B-LOC O O B-LOC I-LOC O O O O O",
        }

    def load_data(self, data_path, no_labels=False):
        split = data_path["split"]
        data_path = data_path["path"]

        filenames = self.test_filenames
        if split == "dev":
            filenames = self.dev_filenames
        data = []

        for fname in filenames:
            path = Path(data_path) / fname
            with open(path, "r") as reader:
                current_sentence = []
                current_label = []
                for line_idx, line in enumerate(reader):
                    if len(line.strip()) == 0:
                        sentence = " ".join(current_sentence)
                        label = " ".join(current_label)
                        data.append(
                            {"input": sentence, "label": label, "line_number": line_idx}
                        )
                        current_sentence = []
                        current_label = []
                    else:
                        elements = line.strip().split()
                        current_sentence.append(elements[0])
                        current_label.append(elements[1])
        return data
