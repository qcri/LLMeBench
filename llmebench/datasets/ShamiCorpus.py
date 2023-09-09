from llmebench.datasets.dataset_base import DatasetBase
from pathlib import Path

class ShamiDataset(DatasetBase): 
    def __init__(self, **kwargs): 
        super(ShamiDataset, self).__init__(**kwargs) 
    def metadata(): 
        return { 
            "language":"ar", 
            "citation": """ @inproceedings{abu-kwaik-etal-2018-shami,
            title = "{S}hami: A Corpus of {L}evantine {A}rabic Dialects",
            author = "Abu Kwaik, Kathrein  and
            Saad, Motaz  and
            Chatzikyriakidis, Stergios  and
            Dobnik, Simon",
            booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
            month = may,
            year = "2018",
            address = "Miyazaki, Japan",
            publisher = "European Language Resources Association (ELRA)",
            url = "https://aclanthology.org/L18-1576",
        }
"""
        }
    def get_data_sample(self): 
        return {"input": "a sentence", "label": "dialect of sentence"} 
    
    def load_data(self, data_path, no_labels=False): 
        data = []
        filenames= ["Jordanian.txt", "Lebanese.txt", "Palestinian.txt", "Syrian.txt"]
        for name in filenames: 
            path = Path(data_path) / name 
            with open(path, "r") as reader: 
                for line in reader: 
                    sentence = line.strip() 
                    label = name.split(".")[0]
                    data.append( 
                        {"input": sentence, "label": label}
                    )
        return data
