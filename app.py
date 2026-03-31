from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        population = float(request.form['population'])
        household = float(request.form['household_size'])
        commercial = float(request.form['commercial_activity'])
        collection = float(request.form['collection_frequency'])
        past_waste = float(request.form['past_waste'])

        income = request.form['income']
        season = request.form['season']

        income_encoded = encoder['income'].transform([income])[0]
        season_encoded = encoder['season'].transform([season])[0]

        features = np.array([[population, household, income_encoded,
                              commercial, collection, season_encoded,
                              past_waste]])

        prediction = model.predict(features)[0]

        return render_template('index.html',
                               prediction_text=f"Predicted Waste: {round(prediction,2)} tons")

    except Exception as e:
        return str(e)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
