import os

from arabic_llm_benchmark.datasets import ArabicPOSDataset
from arabic_llm_benchmark.models import GPTModel, RandomGPTModel
from arabic_llm_benchmark.tasks import ArabicPOSTask


def config():
    sets = [
        ('egy', 'egy.pos/egy.data_5.test.src-trg.sent'),
        ('glf','glf.pos/glf.data_5.test.src-trg.sent'),
        ('mgr','mgr.pos/mgr.data_5.test.src-trg.sent'),
        ('lev','lev.pos/lev.data_5.test.src-trg.sent'),
        ('msa','WikiNewsTruth.txt'),
        ('XGLUE','XGLUE/ar.test.src-tgt.txt')
    ]
    configs = []
    for name, testset in sets:
        configs.append({
            "name": name,
            "config": {
                "dataset": ArabicPOSDataset,
                "dataset_args": {},
                "task": ArabicPOSTask,
                "task_args": {},
                "model": GPTModel,
                "model_args": {
                    "api_type": "azure",
                    "api_version": "2023-03-15-preview",
                    "api_base": os.environ["AZURE_API_URL"],
                    "api_key": os.environ["AZURE_API_KEY"],
                    "engine_name": os.environ["ENGINE_NAME"],
                    # "class_labels": ["m", "f"],
                    "max_tries": 3,
                },
                "general_args": {
                    "data_path": "data/sequence_tagging_ner_pos_etc/POS/"+testset
                },
            }})
    return configs


def prompt(input_sample):
    return {
        "system_message": "You are an AI assistant that helps people find information.",
        "messages": [
            {
                "sender": "user",
                "text": f'Please provide the POS tags for each word in the input sentence. The input will be a list of words in the sentence. \
                 The output format should be a list of tuples, where each tuple consists of a word from the input text and its \
                 corresponding POS tag label from the tag label set: \
                 ["ABBREV", "ADJ", "ADV", "CASE", "CONJ", "DET", "EMOT", "FOREIGN", "FUT_PART", "HASH", "MENTION", "NEG_PART", "NOUN", \
                 "NSUFF", "NUM", "PART", "PREP", "PROG_PART", "PRON", "PUNC", "URL", "V"].\
                Note: Your response should include only a list of tuples, in the order that the words appear in the input sentence, \
                with each tuple containing the corresponding POS tag label for a word. Input:+: {input_sample}',
            }
        ],
    }


def post_process(response):
    text = response["choices"][0]["text"]
    text = re.sub(r"Here's the segmented sentence in a JSON format:",'',text)
    #print("Pro:",text)
    pattern = r"[\"\']([^\"\']+)[\'\"]: *[\'\"]([^}]+)[\'\"]"
    pattern = r"\(\"([^\"]+)\", \"([^\"]+)\"\)"
    matches = re.finditer(pattern, text)
    results = []
    #print("Res0:",results)
    for m in matches:
        tag = m.group(2)
        ntag = []
        for t in tag.split('+'):
            ntag.append(mapTags[t] if t in mapTags else t)
        results.append('+'.join(ntag))
    #print("Res1:",results)
    return ' '.join(results)
