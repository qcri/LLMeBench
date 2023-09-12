from collections import defaultdict

from llmebench import Benchmark

import json

def main():
    benchmark = Benchmark(benchmark_dir="assets")

    assets = benchmark.find_assets()

    train_dataset_metadata = defaultdict(set)
    test_dataset_metadata = defaultdict(set)

    for asset in assets:
        configs = asset["module"].config()
        if isinstance(configs, dict):
            configs = [{"name": "dummy", "config": configs}]

        asset_name = asset["name"]
        for zap in ["_BLOOMZ", "_GPT4", "_GPT35", "_ZeroShot", "_FewShot", "_mdeberta_v3_base_squad2", "_Intfloat_Multilingual_e5_small", "_Camelbert_da_sentiment"]:
            asset_name = asset_name.replace(zap, "")

        for c in configs:
            config = c["config"]
            dataset_name = config["dataset"].__name__
            # print(asset["name"])

            data_path = config["general_args"]["data_path"]
            if isinstance(data_path, dict):
                if "split" in data_path:
                    assert data_path["split"] == "test"
                    data_path = data_path["path"]
                else:
                    data_path = data_path["sentences_path"]

            train_data_path = None
            if "fewshot" in config["general_args"]:
                train_data_path = config["general_args"]["fewshot"]["train_data_path"]
            if isinstance(train_data_path, dict):
                if "split" in train_data_path:
                    assert train_data_path["split"] == "train" or train_data_path["split"] == "dev"
                    train_data_path = train_data_path["path"]
                else:
                    train_data_path = train_data_path["sentences_path"]

            test_dataset_metadata[dataset_name].add((data_path, asset_name))
            if train_data_path:
                train_dataset_metadata[dataset_name].add((train_data_path, asset_name))

    # print("Test data paths")
    for dataset in test_dataset_metadata:
        print(dataset)
        obj = {}
        if dataset == "SemEval23T3PropagandaDataset":
            obj = {}
            mapping = {
                "ar": "ar",
                "en": "en",
                "fr": "fr",
                "ge": "de",
                "it": "it",
                "po": "pl",
                "ru": "ru"
            }
            exceptions = []
            for xlang in mapping:
                lang = mapping[xlang]
                try:
                    obj[lang] = {}
                    test_path = [p for p, _ in test_dataset_metadata[dataset] if f"/{xlang}_" in p][0]
                    if "dev" in test_path:
                        obj[lang]["dev"] = test_path
                    else:
                        obj[lang]["test"] = test_path

                    train_path = [p for p, _ in train_dataset_metadata[dataset] if f"/{xlang}_" in p][0]
                    if "dev" in train_path:
                        assert "dev" not in obj
                        obj[lang]["dev"] = train_path
                    else:
                        obj[lang]["train"] = train_path
                except:
                    exceptions.append(xlang)

            for xlang in exceptions:
                if xlang == "ar":
                    obj["ar"] = {
                        "dev": obj["en"]["dev"][:obj["en"]["dev"].rfind("/")] + "/ar_dev_subtask3.json",
                        "train": obj["en"]["train"][:obj["en"]["train"].rfind("/")] + "/ar_train_subtask3.json"
                    }
                else:
                    raise Exception()
        elif dataset == "CT22CheckworthinessDataset":
            obj = {}
            mapping = {'dutch': "nl",
             'arabic': "ar",
             'spanish': "es",
             'bulgarian': "bg",
             'turkish': "tr",
             'english': "en"
             }
            exceptions = []
            for xlang in mapping:
                lang = mapping[xlang]
                obj[lang] = {}
                try:
                    test_path = [p for p, _ in test_dataset_metadata[dataset] if f"/{xlang}/" in p][0]
                    if "dev" in test_path:
                        obj[lang]["dev"] = test_path
                    else:
                        obj[lang]["test"] = test_path

                    train_path = [p for p, _ in train_dataset_metadata[dataset] if f"/{xlang}/" in p][0]
                    if "dev" in train_path:
                        assert "dev" not in obj
                        obj[lang]["dev"] = train_path
                    else:
                        obj[lang]["train"] = train_path
                except:
                    exceptions.append(xlang)
            assert len(exceptions) == 0
        elif dataset == "AraBenchDataset":
            obj = {
                "APT-LEV": {
                    "train": ("data/MT/ldc_web_lv.train.lev.0.lv.ar", "data/MT/ldc_web_lv.train.lev.0.lv.en"),
                    "dev": ("data/MT/ldc_web_lv.dev.lev.0.lv.ar", "data/MT/ldc_web_lv.dev.lev.0.lv.en"),
                    "test": ("data/MT/ldc_web_lv.test.lev.0.lv.ar", "data/MT/ldc_web_lv.test.lev.0.lv.en")
                },
                "APT-Nile": {
                    "train": ("data/MT/ldc_web_eg.train.nil.0.eg.ar", "data/MT/ldc_web_eg.train.nil.0.eg.en"),
                    "dev": ("data/MT/ldc_web_eg.dev.nil.0.eg.ar", "data/MT/ldc_web_eg.dev.nil.0.eg.en"),
                    "test": ("data/MT/ldc_web_eg.test.nil.0.eg.ar", "data/MT/ldc_web_eg.test.nil.0.eg.en")
                },
                "MADAR-Gulf" : {
                    "train": ("data/MT/madar.train.glf.0.qa.ar", "data/MT/madar.train.glf.0.qa.en"),
                    "dev": ("data/MT/madar.dev.glf.0.qa.ar", "data/MT/madar.dev.glf.0.qa.en"),
                    "test": [
                        ("data/MT/madar.test.glf.0.iq.ar", "data/MT/madar.test.glf.0.iq.en"),
                        ("data/MT/madar.test.glf.1.iq.ar", "data/MT/madar.test.glf.1.iq.en"),
                        ("data/MT/madar.test.glf.2.iq.ar", "data/MT/madar.test.glf.2.iq.en"),
                        ("data/MT/madar.test.glf.0.om.ar", "data/MT/madar.test.glf.0.om.en"),
                        ("data/MT/madar.test.glf.0.qa.ar", "data/MT/madar.test.glf.0.qa.en"),
                        ("data/MT/madar.test.glf.0.sa.ar", "data/MT/madar.test.glf.0.sa.en"),
                        ("data/MT/madar.test.glf.1.sa.ar", "data/MT/madar.test.glf.1.sa.en"),
                        ("data/MT/madar.test.glf.0.ye.ar", "data/MT/madar.test.glf.0.ye.en")
                    ]
                },
                "MADAR-LEV" : {
                    "train": ("data/MT/madar.train.lev.0.lb.ar", "data/MT/madar.train.lev.0.lb.en"),
                    "dev": ("data/MT/madar.dev.lev.0.lb.ar", "data/MT/madar.dev.lev.0.lb.en"),
                    "test": [
                        ("data/MT/madar.test.lev.0.jo.ar", "data/MT/madar.test.lev.0.jo.en"),
                        ("data/MT/madar.test.lev.1.jo.ar", "data/MT/madar.test.lev.1.jo.en"),
                        ("data/MT/madar.test.lev.0.lb.ar", "data/MT/madar.test.lev.0.lb.en"),
                        ("data/MT/madar.test.lev.0.pa.ar", "data/MT/madar.test.lev.0.pa.en"),
                        ("data/MT/madar.test.lev.0.sy.ar", "data/MT/madar.test.lev.0.sy.en"),
                        ("data/MT/madar.test.lev.1.sy.ar", "data/MT/madar.test.lev.1.sy.en")
                    ]
                },
                "MADAR-MGR" : {
                    "train": [
                        ("data/MT/madar.train.mgr.0.ma.ar", "data/MT/madar.train.mgr.0.ma.en"),
                        ("data/MT/madar.train.mgr.0.tn.ar", "data/MT/madar.train.mgr.0.tn.en")
                    ],
                    "dev": ("data/MT/madar.dev.mgr.0.ma.ar", "data/MT/madar.dev.mgr.0.ma.en"),
                    "test": [
                        ("data/MT/madar.test.mgr.0.dz.ar", "data/MT/madar.test.mgr.0.dz.en"),
                        ("data/MT/madar.test.mgr.0.ly.ar", "data/MT/madar.test.mgr.0.ly.en"),
                        ("data/MT/madar.test.mgr.1.ly.ar", "data/MT/madar.test.mgr.1.ly.en"),
                        ("data/MT/madar.test.mgr.0.ma.ar", "data/MT/madar.test.mgr.0.ma.en"),
                        ("data/MT/madar.test.mgr.1.ma.ar", "data/MT/madar.test.mgr.1.ma.en"),
                        ("data/MT/madar.test.mgr.0.tn.ar", "data/MT/madar.test.mgr.0.tn.en"),
                        ("data/MT/madar.test.mgr.1.tn.ar", "data/MT/madar.test.mgr.1.tn.en")
                    ]
                },
                "MADAR-MSA" : {
                    "train": ("data/MT/madar.train.msa.0.ms.ar", "data/MT/madar.train.msa.0.ms.en"),
                    "dev": ("data/MT/madar.dev.msa.0.ms.ar", "data/MT/madar.dev.msa.0.ms.en"),
                    "test": ("data/MT/madar.test.msa.0.ms.ar", "data/MT/madar.test.msa.0.ms.en")
                },
                "MADAR-Nile" : {
                    "train": ("data/MT/madar.train.nil.0.eg.ar", "data/MT/madar.train.nil.0.eg.en"),
                    "dev": ("data/MT/madar.dev.nil.0.eg.ar", "data/MT/madar.dev.nil.0.eg.en"),
                    "test": [
                        ("data/MT/madar.test.nil.0.eg.ar", "data/MT/madar.test.nil.0.eg.en"),
                        ("data/MT/madar.test.nil.1.eg.ar", "data/MT/madar.test.nil.1.eg.en"),
                        ("data/MT/madar.test.nil.2.eg.ar", "data/MT/madar.test.nil.2.eg.en"),
                        ("data/MT/madar.test.nil.0.sd.ar", "data/MT/madar.test.nil.0.sd.en")
                    ]
                },
                "MDC-LEV" : {
                    "dev": ("data/MT/ldc_web_eg.dev.lev.0.sy.ar", "data/MT/ldc_web_eg.dev.lev.0.sy.en"),
                    "test": [
                        ("data/MT/ldc_web_eg.test.lev.0.jo.ar", "data/MT/ldc_web_eg.test.lev.0.jo.en"),
                        ("data/MT/ldc_web_eg.test.lev.0.ps.ar", "data/MT/ldc_web_eg.test.lev.0.ps.en"),
                        ("data/MT/ldc_web_eg.test.lev.0.sy.ar", "data/MT/ldc_web_eg.test.lev.0.sy.en"),
                    ]
                },
                "MDC-MGR" : {
                    "test": ("data/MT/ldc_web_eg.test.mgr.0.tn.ar", "data/MT/ldc_web_eg.test.mgr.0.tn.en"),
                },
                "MDC-MSA" : {
                    "test": ("data/MT/ldc_web_eg.test.msa.0.ms.ar", "data/MT/ldc_web_eg.test.msa.0.ms.en"),
                },
                "Media-Gulf" : {
                    "test": ("data/MT/summa-Oman.test.glf.0.om.ar", "data/MT/summa-Oman.test.glf.0.om.en")
                },
                "Media-LEV" : {
                    "test": ("data/MT/summa-LBC.test.lev.0.lb.ar", "data/MT/summa-LBC.test.lev.0.lb.en")
                },
                "Media-MGR" : {
                    "test": ("data/MT/summa-2M.test.mgr.0.ma.ar", "data/MT/summa-2M.test.mgr.0.ma.en")
                },
                "Media-MSA" : {
                    "test": [
                        ("data/MT/summa-AJ.test.msa.0.ms.ar", "data/MT/summa-AJ.test.msa.0.ms.en"),
                        ("data/MT/summa-BBC.test.msa.0.ms.ar", "data/MT/summa-BBC.test.msa.0.ms.en")
                    ]
                },
                "QAraC-Gulf" : {
                    "dev": ("data/MT/QAraC.dev.glf.0.qa.ar", "data/MT/QAraC.dev.glf.0.qa.ar"),
                    "test": ("data/MT/QAraC.test.glf.0.qa.ar", "data/MT/QAraC.test.glf.0.qa.en")
                },
                "Bible-MGR" : {
                    "train": [
                        ("data/MT/bible.train.mgr.0.ma.ar", "data/MT/bible.train.mgr.0.ma.en"), 
                        ("data/MT/bible.train.mgr.0.tn.ar", "data/MT/bible.train.mgr.0.tn.en"), 
                    ],
                    "dev": [
                        ("data/MT/bible.dev.mgr.0.ma.ar", "data/MT/bible.dev.mgr.0.ma.en"), 
                        ("data/MT/bible.dev.mgr.0.tn.ar", "data/MT/bible.dev.mgr.0.tn.en"), 
                    ],
                    "test": [
                        ("data/MT/bible.test.mgr.0.ma.ar", "data/MT/bible.test.mgr.0.ma.en"), 
                        ("data/MT/bible.test.mgr.0.tn.ar", "data/MT/bible.test.mgr.0.tn.en"),   
                    ]
                },
                "Bible-MSA" : {
                    "train": [
                        ("data/MT/bible.train.msa.0.ms.ar", "data/MT/bible.train.msa.0.ms.en"),
                        ("data/MT/bible.train.msa.1.ms.ar", "data/MT/bible.train.msa.1.ms.en"),
                    ],
                    "dev": [
                        ("data/MT/bible.dev.msa.0.ms.ar", "data/MT/bible.dev.msa.0.ms.en"),
                        ("data/MT/bible.dev.msa.1.ms.ar", "data/MT/bible.dev.msa.1.ms.en")
                    ],
                    "test": [
                        ("data/MT/bible.test.msa.0.ms.ar", "data/MT/bible.test.msa.0.ms.en"),
                        ("data/MT/bible.test.msa.1.ms.ar", "data/MT/bible.test.msa.1.ms.en")
                    ]
                },
                "default": ["APT-LEV", "APT-Nile", "MADAR-Gulf", "MADAR-LEV", "MADAR-MGR", "MADAR-MSA", "MADAR-Nile", "MDC-LEV", "MDC-MGR", "MDC-MSA", "Bible-MGR", "Bible-MSA"]
            }
        elif dataset == "QCRIDialectalArabicSegmentationDataset":
            obj = {
                "test": [
                    "data/sequence_tagging_ner_pos_etc/segmentation/glf.seg/glf.data_5.test.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/lev.seg/lev.data_5.test.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/egy.seg/egy.data_5.test.src.sent",
                    "data/sequence_tagging_ner_pos_etc/segmentation/mgr.seg/mgr.data_5.test.src.sent"
                ]
            }
        elif dataset == "BibleMaghrebiDiacritizationDataset":
            obj = {
                "test": [
                    "data/sequence_tagging_ner_pos_etc/diacritization/morrocan_f05.test.src-tgt.txt",
                    "data/sequence_tagging_ner_pos_etc/diacritization/tunisian_f05.test.src-tgt.txt"
                ]
            }
        elif dataset == "QCRIDialectalArabicPOSDataset":
            obj = {
                "dev": [
                    "data/sequence_tagging_ner_pos_etc/POS/egy.pos/egy.data_5.dev.src-trg.sent",
                    "data/sequence_tagging_ner_pos_etc/POS/glf.pos/glf.data_5.dev.src-trg.sent",
                    "data/sequence_tagging_ner_pos_etc/POS/mgr.pos/mgr.data_5.dev.src-trg.sent",
                    "data/sequence_tagging_ner_pos_etc/POS/lev.pos/lev.data_5.dev.src-trg.sent"
                ],
                "test": [
                    "data/sequence_tagging_ner_pos_etc/POS/egy.pos/egy.data_5.test.src-trg.sent",
                    "data/sequence_tagging_ner_pos_etc/POS/glf.pos/glf.data_5.test.src-trg.sent",
                    "data/sequence_tagging_ner_pos_etc/POS/mgr.pos/mgr.data_5.test.src-trg.sent",
                    "data/sequence_tagging_ner_pos_etc/POS/lev.pos/lev.data_5.test.src-trg.sent"
                ]
            }
        elif len(list(test_dataset_metadata[dataset])) > 1 or len(list(train_dataset_metadata[dataset])) > 1:
            for path, source in train_dataset_metadata[dataset]:
                print(f"\t{path} ({source})")
            for path, source in test_dataset_metadata[dataset]:
                print(f"\t{path} ({source})")
        else:
            test_path = list(test_dataset_metadata[dataset])[0][0]
            if "dev" in test_path:
                obj["dev"] = test_path
            else:
                obj["test"] = test_path

            if dataset in train_dataset_metadata and len(train_dataset_metadata[dataset]) > 0:
                train_path = list(train_dataset_metadata[dataset])[0][0]
                if "dev" in train_path:
                    assert "dev" not in obj
                    obj["dev"] = train_path
                else:
                    obj["train"] = train_path
        print(json.dumps(obj, indent=2))



if __name__ == '__main__':
    main()