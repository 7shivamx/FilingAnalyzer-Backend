# !pip install pdfminer
# !pip install -U transformers
# !python -m nltk.downloader punkt
# !pip install spacy==2.3.5
# !pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz
# !git clone https://github.com/patil-suraj/question_generation.git
# %cd question_generation

import re
import requests
from bs4 import BeautifulSoup as bs
import unicodedata
import sys

import requests
import re
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import spacy

from pipelines import pipeline
from nltk.tokenize import sent_tokenize
import re

from sec_api import FullTextSearchApi
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract metric of the company')
    parser.add_argument('-a', '--api_key', type=str,
                        help='API key for the SEC API')
    parser.add_argument('-m', '--metric', type=str,
                        help='Metric to search for')
    parser.add_argument('--cik', type=str, help='CIK of the company')
    parser.add_argument('-s', '--startDate', type=str,
                        default='2021-01-01', help='Start date of the search')
    parser.add_argument('-e', '--endDate', type=str,
                        default='2022-03-10', help='End date of the search')
    args = parser.parse_args()
    return args


def parse_10q_filing(link, section):
    if section not in [0, 1, 2, 3]:
        print("Not a valid section")
        sys.exit()

    def get_text(link):
        page = requests.get(link, headers={'User-Agent': 'Mozilla'})
        html = bs(page.content, "lxml")
        text = html.get_text()
        text = unicodedata.normalize("NFKD", text).encode(
            'ascii', 'ignore').decode('utf8')
        text = text.split("\n")
        text = " ".join(text)
        return(text)

    def extract_text(text, item_start, item_end):
        item_start = item_start
        item_end = item_end
        starts = [i.start() for i in item_start.finditer(text)]
        ends = [i.start() for i in item_end.finditer(text)]
        positions = list()
        for s in starts:
            control = 0
            for e in ends:
                if control == 0:
                    if s < e:
                        control = 1
                        positions.append([s, e])
        item_length = 0
        item_position = list()
        for p in positions:
            if (p[1]-p[0]) > item_length:
                item_length = p[1]-p[0]
                item_position = p

        item_text = text[item_position[0]:item_position[1]]

        return(item_text)

    text = get_text(link)

    if section == 1 or section == 0:
        try:
            item1_start = re.compile(
                "item\s*[1][\.\;\:\-\_]*\s*\\bF", re.IGNORECASE)
            item1_end = re.compile(
                "item\s*2[\.\;\:\-\_]\s*Man|item\s*3[\.\,\;\:\-\_]\s*Quanti", re.IGNORECASE)
            finText = extract_text(text, item1_start, item1_end)
        except:
            finText = "Something went wrong!"

    if section == 2 or section == 0:
        try:
            item2_start = re.compile(
                "item\s*[2][\.\;\:\-\_]*\s*\\bM", re.IGNORECASE)
            item2_end = re.compile(
                "item\s*3[\.\;\:\-\_]\sQuanti|item\s*4[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            mdaText = extract_text(text, item2_start, item2_end)
        except:
            mdaText = "Something went wrong!"

    if section == 3 or section == 0:
        try:
            item3_start = re.compile(
                "item\s*[3][\.\;\:\-\_]*\s*\\bQ", re.IGNORECASE)
            item3_end = re.compile(
                "item\s*4[\.\;\:\-\_]\sCon|item\s*4[\.\,\;\:\-\_]\s*", re.IGNORECASE)
            riskText = extract_text(text, item3_start, item3_end)
        except:
            riskText = "Something went wrong!"

    if section == 0:
        data = [finText, mdaText, riskText]
    elif section == 1:
        data = [finText]
    elif section == 2:
        data = [mdaText]
    elif section == 3:
        data = [riskText]
    return data


def parse_10k_filling(filing_url):
    api_key = 'd6215192cce355df08e7cf2eaa66d66a01301931447953f2d6457a5d3d82828f'

    response = requests.get(
        f'https://api.sec-api.io/filing-reader?token={api_key}&type=pdf&url={filing_url}')
    with open('/content/metadata.pdf', 'wb') as f:
        f.write(response.content)

    output_string = StringIO()
    with open('/content/metadata.pdf', 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        out = []
        toc_page = -1
        for idx, page in enumerate(PDFPage.create_pages(doc)):
            interpreter.process_page(page)
            out.append(output_string.getvalue().replace('\x00', ''))
            output_string.truncate(0)
            if toc_page == -1 and "table of contents" in out[-1].lower() and 'item' in out[-1].lower():
                toc_page = idx

    if toc_page != -1:
        items_list = re.findall(
            "item +([0-9]+[a-z]?)", out[toc_page].lower().replace('\n', ' '))
    else:
        toc_page = 2

    output = '\n\n'.join(out)
    zero_idx = len('\n\n'.join(out[:toc_page+1])) + 2
    idx = output[zero_idx:].lower().find('item ' + items_list[0] + '.')
    if idx == -1:
        idx = output[zero_idx:].lower().find('item ' + items_list[0])
    if idx == -1:
        idx = output[zero_idx:].lower().find(items_list[0] + '.')

    zero_idx += idx

    output = output[zero_idx:]

    sections_dict = {}
    for item in items_list:
        sections_dict[str(item)] = ''

    last_idx = 0
    last_item = items_list[0]

    for idx, item in enumerate(items_list[1:]):
        cur_output = output[last_idx:].lower()
        cur_idx = cur_output.find('item ' + item + '.')
        # if cur_idx == -1:
        #     cur_idx = cur_output.find('item ' + item)
        if cur_idx == -1 and len(re.findall("[a-z]+", item)):
            cur_idx = cur_output.find(item + '.')
        if cur_idx == -1:
            continue
        sections_dict[last_item] = output[last_idx: last_idx + cur_idx]
        last_idx = last_idx + cur_idx
        last_item = item

    cur_idx = output[last_idx:].lower().find('signature')
    if cur_idx == -1:
        sections_dict[last_item] = output[last_idx:]
    sections_dict[last_item] = output[last_idx: last_idx + cur_idx]
    return sections_dict


def extract_metric(text, metric, val_type="PERCENT", k=8):
    def preprocess_text(text):
        temp_text = text.lower().replace('\n', ' ').replace(' %', '%')
        return temp_text

    def extract_metric_vals(text, val_type="PERCENT", NER=None):
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
            if text_list[idx: idx+len(metric_list)] == metric_list:
                matched_indices.append(idx)
            idx += 1
        return matched_indices

    # 2. get k words before and after the searched metric
    def extract_phrases(text_list, matched_indices, k):
        phrases_extracted = []
        for idx in matched_indices:
            phrase = ""
            for i in range(-k, k+1):
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

    def filter_passage(doc, metric):
        sents = sent_tokenize(doc)
        filtered_sents = ".".join(s for s in sents if metric in s)
        return filtered_sents

    def get_output(passage, question, metric, NER, val_type='PERCENT'):
        tex = preprocess_text(passage)
        filtered_passage = filter_passage(tex, metric)
        ans = nlp({"question": question,  "context": filtered_passage})
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

    NER = spacy.load("en_core_web_sm", disable=[
                     "tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    nlp = pipeline("multitask-qa-qg")

    text = preprocess_text(text)
    possible_values = find_possible_values(text, metric, NER, k, val_type)
    output_values = get_output(
        text, f'What is the value of {metric}?', metric, NER, val_type)
    correct_value = get_correct_value(possible_values, output_values)
    return correct_value


def parse_8k_filing(link):
    page = requests.get(link, headers={'User-Agent': 'Mozilla'})
    html = bs(page.content, "lxml")
    text = html.get_text()
    text = unicodedata.normalize("NFKD", text).encode(
        'ascii', 'ignore').decode('utf8')
    text = text.split("\n")
    text = " ".join(text)
    return text


def get_filings(api_key, metric, cik, startDate='2021-01-01', endDate='2022-03-10'):
    fullTextSearchApi = FullTextSearchApi(api_key=api_key)
    query = {
        "query": f'"{metric.lower()}"',
        "ciks": [cik],
        "formTypes": ['8-K', '10-K', '10-Q'],
        "startDate": startDate,
        "endDate": endDate,
    }
    filings = fullTextSearchApi.get_filings(query)
    return filings


args = parse_args()
metric = args.metric
val_type = 'NUMBER'
filings = get_filings(args.api_key, metric, args.cik,
                      args.startDate, args.endDate)


for filing in filings['filings']:
    if filing['formType'] == '8-K':
        text = parse_8k_filing(filing['filingUrl'])
        value = extract_metric(text, metric, val_type, 8)
    elif filing['formType'] == '10-K':
        filing_10_k_dict = parse_10k_filling(filing['filingUrl'])
        for sec, text in filing_10_k_dict.items():
            value = extract_metric(text, metric, val_type, 8)
            if value:
                break
    elif filing['formType'] == '10-Q':
        filing_10_k_list = parse_10q_filing(filing['filingUrl'], 0)
        for text in filing_10_k_list:
            value = extract_metric(text, metric, val_type, 8)
            if value:
                break
    if value:
        break

print(value)
