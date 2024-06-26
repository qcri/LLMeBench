from enum import Enum

from .Adult import AdultTask
from .ArabicDiacritization import ArabicDiacritizationTask
from .ArabicParsing import ArabicParsingTask
from .ArabicPOS import ArabicPOSTask
from .ArabicSegmentation import ArabicSegmentationTask
from .Attentionworthy import AttentionworthyTask
from .Checkworthiness import CheckworthinessTask
from .ClaimDetection import ClaimDetectionTask
from .Classification import ClassificationTask
from .DemographyGender import DemographyGenderTask
from .DemographyLocation import DemographyLocationTask
from .DemographyNameInfo import DemographyNameInfoTask
from .DialectID import DialectIDTask
from .Emotion import EmotionTask
from .Factuality import FactualityTask
from .HarmfulDetection import HarmfulDetectionTask
from .HateSpeech import HateSpeechTask
from .Lemmatization import LemmatizationTask
from .MachineTranslation import MachineTranslationTask
from .MultilabelPropaganda import MultilabelPropagandaTask
from .MultiNativQA import MultiNativQATask
from .NER import NERTask
from .NewsCategorization import NewsCategorizationTask
from .Offensive import OffensiveTask
from .Q2QSimDetect import Q2QSimDetectionTask
from .QA import QATask
from .Sarcasm import SarcasmTask
from .Sentiment import SentimentTask
from .Spam import SpamTask
from .Stance import StanceTask
from .STS import STSTask
from .Subjectivity import SubjectivityTask
from .XNLI import XNLITask

TaskType = Enum(
    "TaskType",
    [
        "Classification",
        "MultiLabelClassification",
        "SequenceLabeling",
        "QuestionAnswering",
        "SequenceToSequence",
        "Regression",
        "Other",
    ],
)
