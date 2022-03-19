import numpy as np
import re
import argparse

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, help='Text to find sentiment')
    return parser.parse_args()

def get_sentiment(text) :
  input_ids1 = tokenizer(text, truncation=True,padding=True, max_length=128, return_tensors='pt').input_ids
  with torch.no_grad() :
    output = model(input_ids1)['logits']
  probs = softy(output).cpu().numpy()
  return probs

def process_text(document):
  document = re.sub('[^\.A-Za-z0-9]+', ' ', document)
  return document

def get_output(text) :
  processed_text = process_text(text)
  sents = list(processed_text.split('.'))
  filtered_sents = [s.strip() for s in sents if len(s.split()) > 4]
  sentiment = get_sentiment(filtered_sents)
  mean_vals = np.mean(sentiment,axis = 0).tolist()
  infer_dict = {'positive' : mean_vals[0],'negative' : mean_vals[1],'neutral' : mean_vals[2]}
  return infer_dict


args = parse_args()

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

softy = nn.Softmax(dim = -1)

sent_dict = get_output(args.text)
print(sent_dict)