from arabic_llm_benchmark.datasets.dataset_base import DatasetBase
import json
class ArcdDataset(DatasetBase):
    def __init__(self, **kwargs):
        super(ArcdDataset, self).__init__(**kwargs)

    def citation(self):
        return """
                @misc{mozannar2019neural,
            title={Neural Arabic Question Answering}, 
            author={Hussein Mozannar and Karl El Hajal and Elie Maamary and Hazem Hajj},
            year={2019},
            eprint={1906.05394},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
        }
        }"""

    def get_data_sample(self):
        return {"input": {"context": "context for the questions. Usually a snippet of a wikipedia article", 
                          "question": "question to be answered", 
                          "question_id": "a unique question id",
                          "answers": {"text": "answer_text", "answer_start": "where the answer begins"}}, 
                
                "label": "NA", 
                "line_number": "question_number"}

    def load_data(self, data_path, no_labels=False):
        # TODO: modify to iterator
        data = []

        with open(data_path, "r") as reader: 
            dataset = json.load(reader)

        total_quests = 0
        for article in dataset: 
            for paragraph in article["paragraphs"]: 
                context = paragraph["context"] 
                for qa in paragraph["qas"]: 
                    total_quests += 1 
                    question = qa["question"] 
                    question_id = qa["id"]
                    answers = qa["answers"]

                    sample = {"context":context, "question": question, "question_id": question_id, "answers": answers}

                    data.append( 
                        {"input":sample, "label": "NA", "line_number": total_quests }
                    )
        return data