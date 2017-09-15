from flask import Flask, request, render_template
import json
import requests
import socket
from datetime import datetime
import pickle
from model import Fraud
from bs4 import BeautifulSoup
import pandas as pd
from model import Fraud
import time
import pandas as pd
import numpy as np

app = Flask(__name__)



@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/tables.html')
def tables():
    with open('predictions.pkl', 'rb') as pr:
        predictions = pickle.load(pr)
    per_fraud = '{:.2f}% Fraud'.format(np.mean(predictions['prediction']=='Yes')*100)
    data = predictions[['name', 'venue_name', 'org_name', 'prediction']].values
    return render_template('tables.html', data=data, per_fraud=per_fraud)

@app.route('/flot.html')
def flot():
    return render_template('flot.html')

@app.route('/morris.html')
def morris():
    return render_template('morris.html')

@app.route('/forms.html')
def forms():
    return render_template('forms.html')

@app.route('/panels-wells.html')
def panelswells():
    return render_template('panels-wells.html')

@app.route('/buttons.html')
def buttons():
    return render_template('buttons.html')

@app.route('/notifications.html')
def notifications():
    return render_template('notifications.html')

@app.route('/typography.html')
def typography():
    return render_template('typography.html')

@app.route('/icons.html')
def icons():
    return render_template('icons.html')

@app.route('/grid.html')
def grid():
    return render_template('grid.html')

@app.route('/blank.html')
def blank():
    return render_template('blank.html')


@app.route('/login.html')
def login():
    return render_template('login.html')

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080, debug=True)




    pass
