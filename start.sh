#!/bin/bash

source interiit/bin/activate
./interiit/bin/gunicorn --bind 0.0.0.0:5000 wsgi:app --timeout 0
