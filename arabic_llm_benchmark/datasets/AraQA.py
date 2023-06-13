from arabic_llm_benchmark.datasets.dataset_base import DatasetBase
import json

class AraQADataset(DatasetBase):
    def __init__(self, **kwargs):
        super(AraQADataset, self).__init__(**kwargs)

    def citation(self):
        return """ None """
        # ARCD CITATION: 


        # @inproceedings{mozannar-etal-2019-neural,
        # title = \"Neural {A}rabic Question Answering\",
        # author = \"Mozannar, Hussein  and
        # Maamary, Elie  and
        # El Hajal, Karl  and
        # Hajj, Hazem\",
        # booktitle = \"Proceedings of the Fourth Arabic Natural Language Processing Workshop\",
        # month = aug,
        # year = \"2019\",
        # address = \"Florence, Italy\",
        # publisher = \"Association for Computational Linguistics\",
        # url = \"https://www.aclweb.org/anthology/W19-4612\",
        # doi = \"10.18653/v1/W19-4612\",
        # pages = \"108--118\",
        # abstract = \"This paper tackles the problem of open domain factual Arabic question answering (QA) using Wikipedia as our knowledge source. This constrains the answer of any question to be a span of text in Wikipedia. Open domain QA for Arabic entails three challenges: annotated QA datasets in Arabic, large scale efficient information retrieval and machine reading comprehension. To deal with the lack of Arabic QA datasets we present the Arabic Reading Comprehension Dataset (ARCD) composed of 1,395 questions posed by crowdworkers on Wikipedia articles, and a machine translation of the Stanford Question Answering Dataset (Arabic-SQuAD). Our system for open domain question answering in Arabic (SOQAL) is based on two components: (1) a document retriever using a hierarchical TF-IDF approach and (2) a neural reading comprehension model using the pre-trained bi-directional transformer BERT. Our experiments on ARCD indicate the effectiveness of our approach with our BERT-based reader achieving a 61.3 F1 score, and our open domain system SOQAL achieving a 27.6 F1 score.\",
        # }
        
        # MLQA CITATION

        # @article{lewis2019mlqa,
        # title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
        # author={Lewis, Patrick and O\u{g}uz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
        # journal={arXiv preprint arXiv:1910.07475},
        # year={2019}


        # TYDIQA CITATION


        # @article{tydiqa,
        # title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
        # author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
        # year    = {2020},
        # journal = {Transactions of the Association for Computational Linguistics}
        # }

        # XQUAD CITATION


        # @article{Artetxe:etal:2019,
        # author    = {Mikel Artetxe and Sebastian Ruder and Dani Yogatama},
        # title     = {On the cross-lingual transferability of monolingual representations},
        # journal   = {CoRR},
        # volume    = {abs/1910.11856},
        # year      = {2019},
        # archivePrefix = {arXiv},
        # eprint    = {1910.11856}
        # }
        # }
        # """

    def get_data_sample(self):
        return {"context": "جمال أحمد حمزة خاشقجي (13 أكتوبر 1958، المدينة المنورة - 2 أكتوبر 2018)، صحفي وإعلامي سعودي، رأس عدّة مناصب لعدد من الصحف في السعودية، وتقلّد منصب مستشار، كما أنّه مدير عام قناة العرب الإخبارية سابقًا.", 
                "question": " - من هو جمال أحمد حمزة خاشقجي؟ ", 
                "answer": "صحفي وإعلامي"}

    def load_data(self, data_path, no_labels=False):
        """The dataset is save in the SQUAD format just load it as is"""
       
        with open(data_path, "r") as reader: 
            dataset = json.load(reader)["data"]

        # for article in dataset: 
        #     for paragraph in article["paragraphs"]: 
        #         context = paragraph["context"]
        #         for qa in paragraph["qas"]: 
        #             question = qa["question"] 
        #             quest_id = qa["id"] 
        #             data.append({ 
                        
        #             })

        
        
        return dataset
