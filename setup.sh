#!/bin/bash

python3 -m venv interiit
source interiit/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_web_lg
