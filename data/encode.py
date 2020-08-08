import json
from typing import Any, Text, Dict, List

from bert_serving.client import BertClient
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import numpy as np

def encode_standard_question():
    print("in")
    bc = BertClient()
    print("call bc")
    data = json.load(open("./nlu/faq.json", "rt", encoding="utf-8"))
    standard_questions = [each['q'] for each in data]
    print("Standard question size", len(standard_questions))
    print("Start to calculate encoder....")
    standard_questions_encoder = bc.encode(standard_questions)
    np.save("./standard_questions", standard_questions_encoder)
    standard_questions_encoder_len = np.sqrt(np.sum(standard_questions_encoder * standard_questions_encoder, axis=1))
    np.save("./standard_questions_len", standard_questions_encoder_len)


if __name__ == '__main__':
    encode_standard_question()