from flask import Flask, request, render_template
from flask_cors import CORS
from flask_pymongo import PyMongo
import os
import json

# Initialize Flask
app = Flask(__name__, static_url_path='/static', template_folder='static')
CORS(app)

# MongoDB setup
app.config['MONGO_DBNAME'] = 'interiit'
app.config['MONGO_URI'] = 'mongodb://interiit.oz53j.mongodb.net/interiit'
mongo = PyMongo(app)

@app.route('/')
def index():
    return render_template('index.html')
