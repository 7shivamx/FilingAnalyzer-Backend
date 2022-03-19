# !pip install sec-api

from sec_api import ExtractorApi

import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--api_key', type=str, help='API key for sec-api')
    parser.add_argument('-f', '--filing_url', type=str, help='Filing URL')
    parser.add_argument('-s', '--section', type=str, help='Section number')
    return parser.parse_args()

def get_section(filing_url, section = '1'):
    res = filing_url[:filing_url.index("htm") + len("htm")]
    section_text = extractorApi.get_section(str(res), section, "html")
    return section_text

def read_and_clean_table(text):
    tables = pd.read_html(text)
    cleaned_tables = []
    for table in tables:
        table = table.dropna(how='all')
        drop_cols = []
        for col in table.columns:
            nan_cnt = table[col].isnull().sum()
            if nan_cnt >= table.shape[0]/2:
                drop_cols.append(col)
        table = table.drop(labels=drop_cols, axis=1)
        cleaned_tables.append(table)
    return cleaned_tables

def table_to_str(table):
    table = table.values.tolist()
    table_str = []
    for row in table:
        row_cl = [str(data).replace('\xa0', ' ').replace('\\x', ' ').replace('(',' ').replace(')',' ') for data in row]
        table_str.append(' | '.join(row_cl))
    table_str = ' \n '.join(table_str)
    return table_str


args = parse_args()
extractorApi = ExtractorApi(args.api_key)
tables = []

for table in read_and_clean_table('<html><body>' + get_section(args.filing_url, args.section) + '</body></html>'):
    table_data = table_to_str(table)
    tables.append(table_data)

print(tables)
