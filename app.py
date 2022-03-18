from flask import Flask, request, render_template, abort, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from question_generation.pipelines import pipeline
from nltk.tokenize import sent_tokenize
from sec_api import ExtractorApi, QueryApi
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import re
import os
import json
import spacy
import requests

# Initialize Flask
app = Flask(__name__, static_url_path='/static', template_folder='static')
CORS(app)

# MongoDB setup
app.config['MONGO_DBNAME'] = 'interiit'
app.config['MONGO_URI'] = 'mongodb+srv://interiit:interiit@interiit.oz53j.mongodb.net/interiit'
mongo = PyMongo(app)

with open('dict-data.json', 'r') as f:
  data = json.load(f)

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
print("Loading Model...")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

print("Loading NLP...")
NER = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
nlp = pipeline("multitask-qa-qg")

print("Loading Spacy...")
sp = spacy.load('en_core_web_lg')
all_stopwords = sp.Defaults.stop_words


print("Server ready")

metric_val_type = {
    "churn rate": "PERCENT",
    "revenue retention": "PERCENT",
    "LTV to CAC ratio": "RATIO",
    "Customer Engagement Score": "NUMBER",
    "Recurring Revenue": "PERCENT",
    "SAAS Quick Ratio": "RATIO",
    "SAAS Magic Number": "NUMBER"
}
softy = nn.Softmax(dim = -1)

def get_section(filing_url, section) :
    res = filing_url[:filing_url.index("htm") + len("htm")]
    global extractorApi
    section_text = extractorApi.get_section(res, section, "text")
    return section_text

def preprocess_text(text) :
    temp_text = text.lower().replace("\n", " ").replace(' %','%')
    return temp_text

def extract_metric_vals(text, val_type = "PERCENT", NER = None):
    if val_type == "PERCENT":
        return re.findall(r'(\d+(?:\.\d+)?%?(?!\S))', text)
    if val_type == "NUMBER":
        return re.findall(r"[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)", text)
    if val_type == "RATIO":
        return re.findall(r"([0-9]+:[0-9]+)", text)
    if val_type == "MONEY":
        values = []
        entities = NER(text)
        for w in entities.ents:
            if w.label_ == 'MONEY':
                values.append(w.text)
        return values
    return []

# 1. search the metric name in the document
def search_metric(text_list, metric_list):
    matched_indices = []
    idx = 0
    while idx < len(text_list):
        if text_list[idx : idx+len(metric_list)] == metric_list:
            matched_indices.append(idx)
        idx += 1
    return matched_indices

# 2. get k words before and after the searched metric
def extract_phrases(text_list, matched_indices, k):
    phrases_extracted = []
    for idx in matched_indices:
        phrase = ""
        for i in range(-k,k+1):
            if idx+i < 0 or idx+i > len(text_list)-1:
                continue
            phrase += text_list[idx+i] + " "
        phrases_extracted.append(phrase)
    return phrases_extracted

# 3. apply NER and check for corresonding entity
def find_possible_values(text, metric, NER, k, val_type='PERCENT'):
    text = text.replace(',', ' ').replace('-', ' ')
    metric = metric.replace('-', ' ')
    text_list = text.split(' ')
    metric_list = metric.split(' ')
    matched_indices = search_metric(text_list, metric_list)
    phrases_extracted = extract_phrases(text_list, matched_indices, k)
    possible_values = []
    for phrase in phrases_extracted:
        possible_values += extract_metric_vals(phrase, val_type, NER)
    return possible_values

def filter_passage(doc,metric) :
    sents = sent_tokenize(doc)
    filtered_sents = ".".join(s for s in sents if metric in s)
    return filtered_sents

def get_output_for_metrics(passage,question,metric,NER,val_type='PERCENT') :
    tex = preprocess_text(passage)
    filtered_passage = filter_passage(tex,metric)
    ans = nlp({  "question": question,  "context": filtered_passage})
    ans = ans.replace(',', ' ')
    output_values = extract_metric_vals(ans, val_type, NER)
    return output_values

def get_correct_value(possible_values, output_values):
    correct_values = []
    for val1 in output_values:
        for val2 in possible_values:
            if val1 == val2 and val1 not in correct_values:
                correct_values.append(val1)
    if len(correct_values) > 0:
        return correct_values[-1]
    return ''

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

def ratio_to_float(ratio):
    numer,denum = ratio.split(":")
    if int(denum) == 0:
        num = math.inf
    else:
        num = int(numer)/int(denum)
    return num

# For serving frontend
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/dictsent", methods = ["POST"])
def sentimentRequest1():
    req_data = request.get_json()
    try:
        sentence = req_data['q']
        sent_dict = get_class_counter(sentence)
        return jsonify(sent_dict)
    except Exception as e:
        print(e)
        return "Invalid request"

@app.route("/bertinf", methods = ["POST"])
def sentimentRequest2():
    req_data = request.get_json()
    try:
        sentence = req_data['q']
        bert_dict = get_output(sentence)
        return jsonify(bert_dict)
    except Exception as e:
        print(e)
        return "Invalid request"

# CIK, Ticker by name
@app.route("/companybyname", methods = ["POST"])
def companybyname():
    req_data = request.get_json()
    try:
        name = req_data["name"]
        result = mongo.db.targets.find_one({'Name': re.compile('^' + re.escape(name) + '$', re.IGNORECASE)})
        return {"cik": result["CIK"], "ticker": result["Ticker"], "ARR": result["ARR"], "NRR": result["NRR"], "Customers": result["Customers"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Name, Ticker by CIK
@app.route("/companybycik", methods = ["POST"])
def companybycik():
    req_data = request.get_json()
    try:
        cik = req_data["cik"]
        result = mongo.db.targets.find_one({'CIK': cik})
        return {"name": result["Name"], "ticker": result["Ticker"], "ARR": result["ARR"], "NRR": result["NRR"], "Customers": result["Customers"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Name, CIK by ticker
@app.route("/companybyticker", methods = ["POST"])
def companybyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        result = mongo.db.targets.find_one({'Ticker': re.compile('^' + re.escape(ticker) + '$', re.IGNORECASE)})
        return {"name": result["Name"], "cik": result["CIK"], "ARR": result["ARR"], "NRR": result["NRR"], "Customers": result["Customers"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Main Timeseries by ticker
@app.route("/tsbyticker", methods = ["POST"])
def tsbyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        result = mongo.db.targets.find_one({'Ticker': re.compile('^' + re.escape(ticker) + '$', re.IGNORECASE)})
        return {"arrTS": result["arrTS"], "nrrTS": result["nrrTS"], "custTS": result["custTS"], "quarTS": result["quarTS"], "smTS": result["smTS"], "empTS": result["empTS"], "srcTS": result["srcTS"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# QnA and Summary by ticker
@app.route("/qnabyticker", methods = ["POST"])
def qnabyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        result = mongo.db.targets.find_one({'Ticker': re.compile('^' + re.escape(ticker) + '$', re.IGNORECASE)})
        return {"qna": result["qna"], "summary": result["summary"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Twitter trending words by ticker
@app.route("/twitbyticker", methods = ["POST"])
def twitbyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        result = mongo.db.targets.find_one({'Ticker': re.compile('^' + re.escape(ticker) + '$', re.IGNORECASE)})
        return {"trendingWords": result["trendingWords"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Dict and Finbert sentiments of last 10k by ticker
@app.route("/sentibyticker", methods = ["POST"])
def sentibyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        result = mongo.db.targets.find_one({'Ticker': re.compile('^' + re.escape(ticker) + '$', re.IGNORECASE)})
        return {"dictSenti": result["dictSenti"], "finbSenti": result["finbSenti"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Sectionwise data of last 10k by ticker
@app.route("/secbyticker", methods = ["POST"])
def secbyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        result = mongo.db.targets.find_one({'Ticker': re.compile('^' + re.escape(ticker) + '$', re.IGNORECASE)})
        return {"sectionwise": result["sectionwise"]}
    except Exception as e:
        print(e)
        return "Invalid request"

# Company overview by ticker (Current pe ratio, eps, operating margin(TTM), etc)
@app.route("/overviewbyticker", methods = ["POST"])
def overviewbyticker():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        req = requests.get(url = "https://www.alphavantage.co/query", params = {"function":"OVERVIEW", "apikey":"PQCT0KTQ95W1SK9W", "symbol":ticker})
        data = req.json()
        return {
            "description":data["Description"],
            "exchange":data["Exchange"],
            "quater":data["LatestQuarter"],
            "pe":data["PERatio"],
            "divi":data["DividendPerShare"],
            "eps":data["EPS"],
            "profitmargin":data["ProfitMargin"],
            "operatingmarginttm":data["OperatingMarginTTM"]
        }
    except Exception as e:
        print(e)
        return "Invalid request"

# Timeseries of income statement metrics (gross profit margin %, operating expenses) by ticker, size of timeperiod
@app.route("/incometimeseries", methods = ["POST"])
def incometimeseries():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        timeperiod = req_data["timeperiod"]
        req = requests.get(url = "https://www.alphavantage.co/query", params = {"function":"INCOME_STATEMENT", "apikey":"PQCT0KTQ95W1SK9W", "symbol":ticker})
        data = req.json()
        if timeperiod == "annual":
            data = data["annualReports"]
        else:
            data = data["quarterlyReports"]
        result = []
        for x in data:
            gpm_val = float(x["grossProfit"])/float(x["totalRevenue"])*100
            if gpm_val >= 75:
                gpm_status = "good"
            if gpm_val <= 70:
                gpm_status = "bad"
            result.append({"date": x["fiscalDateEnding"], "opex": x["operatingExpenses"], "gpm": gpm_val, "condition": gpm_status})
        return {"data": result}
    except Exception as e:
        print(e)
        return "Invalid request"

# Timeseries of EPS by ticker, size of timeperiod
@app.route("/earningstimeseries", methods = ["POST"])
def earningstimeseries():
    req_data = request.get_json()
    try:
        ticker = req_data["ticker"]
        timeperiod = req_data["timeperiod"]
        req = requests.get(url = "https://www.alphavantage.co/query", params = {"function":"EARNINGS", "apikey":"PQCT0KTQ95W1SK9W", "symbol":ticker})
        data = req.json()
        if timeperiod == "annual":
            data = data["annualEarnings"]
        else:
            data = data["quarterlyEarnings"]
        result = []
        for x in data:
            result.append({"date": x["fiscalDateEnding"], "eps": x["reportedEPS"]})
        return {"data": result}
    except Exception as e:
        print(e)
        return "Invalid request"

@app.route("/extract", methods = ["POST"])
def extractRequest():
    req_data = request.get_json()
    try:
        api_key = "b6b10b35e011e60a5e028ba927a6ae3223eca8389d7b987c93bf59f9a00f26de"
        timeperiod = req_data['timeperiod'].lower()
        cik = req_data['cik']
        from_date = req_data['from_date']
        to_date = req_data['to_date']

        global extractorApi
        extractorApi = ExtractorApi(api_key)
        global queryApi
        queryApi = QueryApi(api_key = api_key)

        metric = req_data['metric']
        val_type = metric_val_type[metric].upper()
        k = 6
        relevant_sections = ['1', '2','3','4','5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        type_of_form = "10-K"

        if timeperiod == "annual":
            type_of_form = "10-K"
        else:
            type_of_form = "10-Q"


        query = {
            "query": { "query_string": {
                "query": f"cik:{cik} AND filedAt:{{{from_date} TO {to_date}}} AND formType:\"{type_of_form}\""
                } },
            "from": "0",
            "size": "10",
            "sort": [{ "filedAt": { "order": "desc" } }]
            }

        return_list = queryApi.get_filings(query)
        values = []

        for filing in return_list['filings']:
            no_value = True
            filedAt = filing['filedAt']
            filedAt = filedAt[:filedAt.index('T')]

            for sec in relevant_sections:
                text = get_section(filing['linkToFilingDetails'], sec)
                text = preprocess_text(text)
                possible_values = find_possible_values(text, metric, NER, int(k), val_type)
                output_values = get_output_for_metrics(text, f'What is the value of {metric}?', metric, NER, val_type)
                correct_value = get_correct_value(possible_values, output_values)
                if len(correct_value) > 0:
                    no_value = False
                    met_val = correct_value
                    if metric == "churn rate":
                       met_val = float(met_val.strip(' \t\n\r%'))
                       if met_val <= 1:
                           condition = "good"
                       elif met_val >= 2:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    elif metric == "revenue retention":
                       met_val = float(met_val.strip(' \t\n\r%'))
                       if met_val >= 90:
                           condition = "good"
                       elif met_val < 70:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    elif metric == "LTV to CAC ratio":
                       c = ratio_to_float(met_val)
                       if c >= 3:
                           condition = "good"
                       elif c <= 1:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    elif metric == "Customer Engagement Score":
                       if met_val >= 71:
                           condition = "good"
                       elif met_val <= 40:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    elif metric == "Recurring Revenue":
                       met_val = float(met_val.strip(' \t\n\r%'))
                       if met_val >= 10:
                           condition = "good"
                       elif met_val <= 5.7:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    elif metric == "SAAS Quick Ratio":
                       c = ratio_to_float(met_val)
                       if c >= 4:
                           condition = "good"
                       elif c <= 1:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    elif metric == "SAAS Magic Number":
                       if met_val >= 0.75:
                           condition = "good"
                       elif met_val <= 0.50:
                           condition = "bad"
                       else:
                           condition = "neutral"
                    values.append((correct_value, filedAt, condition))
                    break

            if no_value:
                values.append(('-1', filedAt))

        return jsonify({"correct_value" : values}), 200

    except:
        return jsonify({"error": "Post Parameters not provided"}), 400


if __name__ == "__main__":
    app.run(debug=True)
