from flask import Flask, request, render_template, abort, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import numpy as np
import re
import os
import json
import spacy

# Initialize Flask
app = Flask(__name__, static_url_path='/static', template_folder='static')
CORS(app)

# MongoDB setup
app.config['MONGO_DBNAME'] = 'interiit'
app.config['MONGO_URI'] = 'mongodb://interiit.oz53j.mongodb.net/interiit'
mongo = PyMongo(app)

with open('./dict-data.json', 'r') as f:
  data = json.load(f)
  
sp = spacy.load('en_core_web_lg')
all_stopwords = sp.Defaults.stop_words

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

softy = nn.Softmax(dim = -1)

def get_class_counter(document) :
    doc = preprocess_data(document)
    sentiment_dict = {}
    tokens_temp = doc.split()
    tokens = [t for t in tokens_temp if t not in all_stopwords]
    for token in tokens :
        if token in data :
            for class_label in data[token] :
                if class_label in sentiment_dict :
                    sentiment_dict[class_label] += 1
                else :
                    sentiment_dict[class_label] = 1   
    for key in sentiment_dict :
        sentiment_dict[key] = float(sentiment_dict[key]/len(tokens))
    return sentiment_dict
    
def preprocess_data(doc) :
  document = re.sub('[^A-Za-z0-9]+', ' ', doc)
  document = document.lower()   
  return document 

def get_sentiment(text) :
  input_ids1 = tokenizer(text, truncation=True,padding=True, max_length=128, return_tensors='pt').input_ids
  with torch.no_grad() :
    output = model(input_ids1)['logits']
  probs = softy(output).cpu().numpy()
  # infer_dict = {'positive' : probs[0],'negative' : probs[1], 'neutral' : probs[2]}
  return probs
  
def process_text(document):
  document = re.sub('[^\.A-Za-z0-9]+', ' ', document)
  # document = document.lower()   
  return document
  
def get_output(text) :
  processed_text = process_text(text)
  sents = list(processed_text.split('.'))
  filtered_sents = [s.strip() for s in sents if len(s.split()) > 4]
  sentiment = get_sentiment(filtered_sents)
  mean_vals = np.mean(sentiment,axis = 0).tolist()
  infer_dict = {'positive' : mean_vals[0],'negative' : mean_vals[1],'neutral' : mean_vals[2]}
  return infer_dict       
  
@app.route('/')
def index():
    return render_template('index.html')  

@app.route("/dictsent", methods = ["GET","POST"])
def sentimentRequest1():
    if request.method == "POST":
        sentence = request.form['q']
        
    else:
        sentence = request.args.get('q')
        
    sent_dict = get_class_counter(sentence)
    return jsonify(sent_dict) 
    
@app.route("/bertinf", methods = ["GET","POST"])
def sentimentRequest2():
    if request.method == "POST":
        sentence = request.form['q']
        
    else:
        sentence = request.args.get('q')
        
    bert_dict = get_output(sentence)
    return jsonify(bert_dict)             

if __name__ == "__main__":
    app.run(debug=True)    
    
    
