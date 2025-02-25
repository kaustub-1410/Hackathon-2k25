from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Breast Cancer Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input JSON
    features = np.array(data['features']).reshape(1, -1)  # Convert to numpy array
    prediction = model.predict(features)  # Make prediction
    return jsonify({'prediction': int(prediction[0])})  # Return response

if __name__ == '__main__':
    app.run(debug=True)
