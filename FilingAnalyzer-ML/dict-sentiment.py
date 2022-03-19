import json
import re
import argparse
import spacy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str,
                        help='Text to find sentiment')
    return parser.parse_args()


def get_class_counter(document):
    doc = preprocess_data(document)
    sentiment_dict = {}
    tokens_temp = doc.split()
    tokens = [t for t in tokens_temp if t not in all_stopwords]
    for token in tokens:
        if token in data:
            for class_label in data[token]:
                if class_label in sentiment_dict:
                    sentiment_dict[class_label] += 1
                else:
                    sentiment_dict[class_label] = 1
    for key in sentiment_dict:
        sentiment_dict[key] = float(sentiment_dict[key]/len(tokens))
    return sentiment_dict


def preprocess_data(doc):
    document = re.sub('[^A-Za-z0-9]+', ' ', doc)
    document = document.lower()
    return document


with open('./dict-data.json', 'r') as f:
    data = json.load(f)

sp = spacy.load('en_core_web_lg')
all_stopwords = sp.Defaults.stop_words

args = parse_args()
sent_dict = get_class_counter(args.text)
print(sent_dict)
