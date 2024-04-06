from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

app = Flask(__name__)

# Load the dictionary mapping symptoms to their indices
df = pd.read_csv('Data/Training.csv')
symptoms_dict = {symptom: index for index, symptom in enumerate(df.columns[:-1])}

# Load the trained model from the .pkl file
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def home():
    return "Hello Nutesh"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the symptom list from the request
    symptoms_exp = request.json.get('symptoms')

    # Create an input vector based on the provided symptoms
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_exp:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    # Make a prediction using the loaded model
    prediction = loaded_model.predict([input_vector])[0]

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction})

