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

    prompt = (
        "You are an expert social media analyst specializing in identifying propaganda in Arabic contexts. "
        "I will provide you with an Arabic meme, and your task is to analyze it step by step before reaching a final conclusion.\n\n"

        "Follow these steps carefully:\n\n"

        "### Step 1: Extract Text from the Meme\n"
        "1. Identify and extract any text present in the meme image.\n"
        "2. If there is no readable text, return an empty string.\n\n"

        "### Step 2: Generate a Description of the Image\n"
        "1. Describe the visual elements in the image in a concise manner (maximum 50 words).\n"
        "2. Focus on key elements such as people, objects, symbols, or emotions conveyed.\n\n"

        "### Step 3: Extract Entity Mentions\n"
        "1. Identify and extract any named entities present in the text or visible in the image (e.g., PERSON, ORGANIZATION, LOCATION).\n"
        "2. If no entities are detected, return an empty list.\n\n"

        "### Step 4: Identify Key Themes and Indicators\n"
        "1. Analyze the extracted text and image description together.\n"
        "2. Identify any potential themes related to propaganda (e.g., nationalism, disinformation, bias, exaggeration, hate speech, manipulation).\n"
        "3. Take into account **both textual and visual elements** to determine the intent behind the meme.\n\n"

        "### Step 5: Classify the Meme\n"
        "Based on the extracted information and themes identified:\n"
        "1. Classify the meme into one of the following categories:\n"
        "   (a) propaganda\n"
        "   (b) not_propaganda\n"
        "   (c) other (if the meme does not clearly fit as propaganda or not_propaganda)\n"
        "   (d) not-meme (if it does not qualify as a meme)\n"
        "2. Explain your reasoning before finalizing the classification.\n\n"

        "### Step 6: Rate Your Confidence Level\n"
        "1. On a scale of 1 to 10, rate how confident you are in your classification decision.\n"
        "2. Justify your confidence level (e.g., was there ambiguity in the meme? Was the propaganda intent unclear?).\n\n"

        "### Step 7: Self-Verification Check\n"
        "1. Review your classification and explanation.\n"
        "2. Ask yourself: **Does my explanation align with my classification?**\n"
        "3. If there is an inconsistency, refine your explanation or reconsider the label.\n\n"

        "### Step 8: Assess Hateful Content\n"
        "1. Using all extracted information ('text', 'description', 'entity mentions', multimodal details, and classification label), determine if the meme is:\n"
        "   (a) hateful\n"
        "   (b) not-hateful\n"
        "2. Provide a **detailed justification** explaining why the meme is or is not hateful.\n\n"

        "### Final Output Format\n"
        "Your response must be formatted as a valid JSON object with the following structure:\n\n"

        "{\n"
        '  "extracted_text": "text",\n'
        '  "description": "short description",\n'
        '  "entity_mentions": ["mention1", "mention2"],\n'
        '  "key_themes": ["theme1", "theme2"],\n'
        '  "propaganda_classification": "propaganda/not_propaganda/other/not-meme",\n'
        '  "propaganda_classification_explanation": "explanation",\n'
        '  "confidence_score": score (1-10),\n'
        '  "confidence_explanation": "explanation",\n'
        '  "hate_label": "hateful/not-hateful",\n'
        '  "hatefulness_justification": "explanation"\n'
        "}"
    )


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
            
        }
    ]


def post_process(response):
    data = response["choices"][0]["message"]["content"]
    data = re.search(r"```json\n(.*)\n```", data, re.DOTALL).group(1)
    data = json.loads(data)

    # Extract the classification label
    classification_label = data["propaganda_classification"]

    return classification_label
