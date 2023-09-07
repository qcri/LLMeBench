from llmebench.datasets.dataset_base import DatasetBase


class BibleMaghrebiDiacritizationDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(BibleMaghrebiDiacritizationDataset, self).__init__(**kwargs)

    def metadata():
        return {
            "language": "ar",
            "citation": """@InProceedings{DARWISH18.20,
                author = {Kareem Darwish , Ahmed Abdelali , Hamdy Mubarak , Younes Samih and Mohammed Attia},
                title = {Diacritization of Moroccan and Tunisian Arabic Dialects: A CRF Approach},
                booktitle = {Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018)},
                year = {2018},
                month = {may},
                date = {7-12},
                location = {Miyazaki, Japan},
                editor = {Hend Al-Khalifa and King Saud University and KSA Walid Magdy and University of Edinburgh and UK Kareem Darwish and Qatar Computing Research Institute and Qatar Tamer Elsayed and Qatar University and Qatar},
                publisher = {European Language Resources Association (ELRA)},
                address = {Paris, France},
                isbn = {979-10-95546-25-2},
                language = {english}
            }""",
        }

    def get_data_sample(self):
        return {
            "input": "Original sentence",
            "label": "Sentence with diacritized words",
        }

    def load_data(self, data_path, no_labels=False):
        data = []

        with open(data_path, "r") as fp:
            for line_idx, line in enumerate(fp):
                text, diacritized_text = line.split("\t")
                data.append(
                    {
                        "input": text.strip(),
                        "label": diacritized_text.strip(),
                        "line_number": line_idx,
                    }
                )

        return data
