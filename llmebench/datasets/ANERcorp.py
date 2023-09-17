from llmebench.datasets.dataset_base import DatasetBase
from llmebench.tasks import TaskType


class ANERcorpDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ANERcorpDataset, self).__init__(**kwargs)

    @staticmethod
    def metadata():
        return {
            "language": "ar",
            "citation": """@InProceedings{10.1007/978-3-540-70939-8_13,
                author="Benajiba, Yassine
                and Rosso, Paolo
                and Bened{\'i}Ruiz, Jos{\'e} Miguel",
                editor="Gelbukh, Alexander",
                title="ANERsys: An Arabic Named Entity Recognition System Based on Maximum Entropy",
                booktitle="Computational Linguistics and Intelligent Text Processing",
                year="2007",
                publisher="Springer Berlin Heidelberg",
                address="Berlin, Heidelberg",
                pages="143--153",
                abstract="The task of Named Entity Recognition (NER) allows to identify proper names as well as temporal and numeric expressions, in an open-domain text. NER systems proved to be very important for many tasks in Natural Language Processing (NLP) such as Information Retrieval and Question Answering tasks. Unfortunately, the main efforts to build reliable NER systems for the Arabic language have been made in a commercial frame and the approach used as well as the accuracy of the performance are not known. In this paper, we present ANERsys: a NER system built exclusively for Arabic texts based-on n-grams and maximum entropy. Furthermore, we present both the specific Arabic language dependent heuristic and the gazetteers we used to boost our system. We developed our own training and test corpora (ANERcorp) and gazetteers (ANERgazet) to train, evaluate and boost the implemented technique. A major effort was conducted to make sure all the experiments are carried out in the same framework of the CONLL 2002 conference. We carried out several experiments and the preliminary results showed that this approach allows to tackle successfully the problem of NER for the Arabic language.",
                isbn="978-3-540-70939-8"
            }
            @inproceedings{obeid-etal-2020-camel,
                title = "{CAM}e{L} Tools: An Open Source Python Toolkit for {A}rabic Natural Language Processing",
                author = "Obeid, Ossama  and
                  Zalmout, Nasser  and
                  Khalifa, Salam  and
                  Taji, Dima  and
                  Oudah, Mai  and
                  Alhafni, Bashar  and
                  Inoue, Go  and
                  Eryani, Fadhl  and
                  Erdmann, Alexander  and
                  Habash, Nizar",
                booktitle = "Proceedings of the Twelfth Language Resources and Evaluation Conference",
                month = may,
                year = "2020",
                address = "Marseille, France",
                publisher = "European Language Resources Association",
                url = "https://aclanthology.org/2020.lrec-1.868",
                pages = "7022--7032",
                abstract = "We present CAMeL Tools, a collection of open-source tools for Arabic natural language processing in Python. CAMeL Tools currently provides utilities for pre-processing, morphological modeling, Dialect Identification, Named Entity Recognition and Sentiment Analysis. In this paper, we describe the design of CAMeL Tools and the functionalities it provides.",
                language = "English",
                ISBN = "979-10-95546-34-4",
            }""",
            "link": "https://camel.abudhabi.nyu.edu/anercorp/",
            "license": "CC BY-SA 4.0",
            "splits": {
                "test": "ANERCorp_CamelLab_test.txt",
                "train": "ANERCorp_CamelLab_train.txt",
            },
            "task_type": TaskType.SequenceLabeling,
            "class_labels": [
                "B-PERS",
                "I-PERS",
                "B-LOC",
                "I-LOC",
                "B-ORG",
                "I-ORG",
                "B-MISC",
                "I-MISC",
                "O",
            ],
        }

    @staticmethod
    def get_data_sample():
        return {
            "input": ".كانت السبب الرئيس في سقوط البيزنطيين بسبب الدمار الذي كانت تخلفه الحملات الأولى المارة في بيزنطة ( مدينة القسطنطينية ) عاصمة الإمبراطورية البيزنطية وتحول حملات لاحقة نحوها",
            "label": "O O O O O B-PER O O O O O O O O O B-LOC O O B-LOC O O B-LOC I-LOC O O O O O",
        }

    def load_data(self, data_path, no_labels=False):
        data_path = self.resolve_path(data_path)
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
                    current_sentence = []
                    current_label = []
                else:
                    elements = line.strip().split()
                    current_sentence.append(elements[0])
                    current_label.append(elements[1])
        return data
