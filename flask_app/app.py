from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load the model (adjust the path as necessary)
model_path = 'C:/Users/Nayeem Islam/Desktop/AirLine Dynamic Profit - ML/notebooks/model/ticket_price_predictor_rf.joblib'
model = load(model_path)

@app.route('/', methods=['GET'])
def home():
    # Serve the HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
