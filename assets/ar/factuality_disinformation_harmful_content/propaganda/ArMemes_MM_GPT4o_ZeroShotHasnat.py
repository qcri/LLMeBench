import json
import re

from llmebench.datasets import ArMemesDataset
from llmebench.models import OpenAIModel
from llmebench.tasks import ClassificationTask


def metadata():
    return {
        "author": "Arabic Language Technologies, QCRI, HBKU",
        "model": "gpt-4o1",
        "description": "GPT-4o1 model hosted on Azure, using the ChatCompletion API. API version '2024-02-15-preview'.",
        "scores": {
            "Micro-F1": "0.5661375661375662",
            "Weighted F1": "0.5296411577439224",
            "Macro F1": "0.2514937653942827",
        },
    }


def config():
    return {
        "dataset": ArMemesDataset,
        "task": ClassificationTask,
        "model": OpenAIModel,
        "model_args": {
            "class_labels": ["not_propaganda", "propaganda", "not-meme", "other"],
            "max_tries": 30,
            "max_tokens":1000,
        },
    }


def prompt(input_sample):
    base64_image = input_sample["image"]
    # prompt = (
    #     "You are an expert social media analyst specializing in identifying propaganda in Arabic contexts. "
    #     "I will provide you with an Arabic meme. You need to perform the following tasks:\n"
    #     "1) Your task is to extract 'text' from the meme.\n"
    #     "2) Generate a 'description' of the image. The 'description' should be very short in maximum 50 words\n"
    #     "3) Extract 'entity mentions' if there are any. For example, PERSON.\n"
    #     "4) Use the 'text', 'description', 'entity mentions' and multimodal information of the meme to classify whether the meme is (a) propaganda, (b) not_propaganda, (c) other, or (d) not-meme. Note: 'other' is a category used to label the image if it cannot be classified as 'propaganda', 'not_propaganda', or 'not-meme'.\n"
    #     "5) Then judge your decision on a scale of 1 to 10.\n"
    #     "6) Justify the decision of the label with an explanation.\n"
    #     "7) If the label you predicted in step 3 and explanation you provided does not match, please verify and the classified label and explanation.\n"
    #     "8) Use 'text', 'description', 'entity mentions', multimodal information and propaganda classified label extracted in step 1, 2, 3, 4 and 6 to identify whether the meme is (a) 'hateful' or (b) 'not-hateful'. Provide a justification the hatefulness\n"
    #     "You should output all extracted information in valid JSON with the format below:\n"
    #     "{\n"
    #     '"extracted_text": "text",\n'
    #     '"description": "description",\n'
    #     '"entity_mentions": "mention1, mention2",\n'
    #     '"classification": "propagandistic",\n'
    #     '"classification_explanation": "explanation",\n'
    #     '"judgement": score,\n'
    #     '"hate_label": hateful,\n'
    #     '"hatefulness_justification": "explanation"\n'
    #     "}"
    # )
    explanation_length=100

    prompt = f'''
    You are a Propaganda Image Detection Expert specialized in analyzing images and memes from social media. Your goal is to determine whether an image is a meme and if it contains propagandistic elements. Follow these steps:
    1. Image Inspection: Examine all visual elements (people, symbols, objects, background, colors, etc.). Note any distinctive or potentially manipulative visuals. 
    2. Text Extraction and Analysis: Identify any text in the image (captions, banners, speech bubbles, etc.). Summarize the text. Check for loaded language, emotionally charged words, or misleading statements.
    3. Contextual Alignment: Assess how the text and visuals interact. Do they reinforce or contradict each other? Determine if the combination of text and imagery suggests a specific agenda or manipulative intent.
    4. Propaganda Indicators: Look for techniques such as:
    - Emotional appeal or fearmongering (targeting specific group fears or sentiments)
    - Exaggeration or distortion of facts
    - Loaded or polarizing language
    - Symbolic use of imagery (e.g., flags, historical figures, strong cultural icons used in a misleading context)
    - Stereotyping or scapegoating of individuals or groups    
    5. Propaganda Classification: Provide one of the following classes:
    - propaganda
    - not-propaganda
    - not-meme: Lacks the defining features of a meme (e.g., humor, overlaid text, viral intent).
    - other: Includes offensive content, non-Arabic text, or unintelligible visuals (e.g., poor font, dialect issues).
    7. Output Requirements: Your answer must be in valid JSON format with the following fields:
    "class": one of ["not-propaganda", "propaganda", "not-meme", "other"]
    "explanation": a concise explanation (in {explanation_length} words or fewer) supporting your classification
    Final Answer Format (Example):
    {{
      "class": "propaganda",
      "explanation": "The image text uses emotionally charged language aligned with the visual to incite fear, suggesting manipulative intent."
    }}
  '''
  # - "not-meme" (if the image contains no text)
  # - "other" (if the content does not fit clearly into the above categories)

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
            "response_format": {
                        "type": "json"
            }, 
        }
    ]


def post_process(response):
    data = response["choices"][0]["message"]["content"]
    data = re.search(r"```json\n(.*)\n```", data, re.DOTALL).group(1)
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["class"]

    return classification_label
