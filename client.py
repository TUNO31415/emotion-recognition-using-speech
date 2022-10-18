from crypt import methods
from urllib import request
from flask import Flask, jsonify, request
import pandas as pd
from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
import datetime

app = Flask(__name__)
rec = EmotionRecognizer(model=SVC(), emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
rec.train()

states =[
    {'date' : '01/11/2022', 'time' : '11:00:19', 'state' : 'happy'}
]

@app.route('/audio')
def hello_world():
    return jsonify(states)

# Get audio somehow from furhat using POST request
@app.route('/audio', methods=['POST'])
def get_audio():
    now = datetime.now()
    date = now.strftime('%H:%M:%S')
    time = now.strftime('%d/%m/%Y')
    prediction = rec.pridict('PATH-TO-AUDIO')

    states = [
        {'date' : date,
        'time' : time,
        'state' : prediction}
    ]
    return f'Prediction : {prediction}', 204

if __name__ == '__main__':
    app.run()