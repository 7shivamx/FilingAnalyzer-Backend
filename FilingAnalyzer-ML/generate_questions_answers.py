# !pip install -U transformers==3.0.0
# !python -m nltk.downloader punkt
# !git clone https://github.com/patil-suraj/question_generation.git
# %cd question_generation

import argparse
from pipelines import pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, help='Text to summarize')
    return parser.parse_args()


args = parse_args()
nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")
qna_dict = nlp(args.text)
print(qna_dict)
