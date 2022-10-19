from flask import Flask, jsonify
from emotion_recognition import EmotionRecognizer
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)
rec = EmotionRecognizer(model=MLPClassifier(alpha=1, max_iter=1000), emotions=['sad', 'neutral', 'happy'], balance=True, verbose=0)
rec.train()

# For test purpose
@app.route('/audio', methods=['GET'])
def get_state():
    prediction = rec.predict_proba("data/emodb/wav/16a01Wb.wav")
    return jsonify(prediction)

# Get audio somehow from furhat using POST request
@app.route('/audio', methods=['POST'])
def get_audio():
    return jsonify(rec.pridict_proba('PATH-TO-AUDIO'))

if __name__ == '__main__':
    app.run()