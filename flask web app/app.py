import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("../ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return {'message': 'This is a web app for ML.'}

# http://127.0.0.1:5000/predict?seconds=50&lat=44&lng=-12

@app.route("/predict", methods=["GET"])
def predict():

    args = request.args
    seconds = int( args['seconds'])
    lat = int( args['lat'])
    lng = int( args['lng'])

    int_features = [seconds, lat, lng]


    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return {'country': countries[output]}


if __name__ == "__main__":
    app.run(debug=True)