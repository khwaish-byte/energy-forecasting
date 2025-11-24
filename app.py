import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from datetime import datetime
app = Flask(__name__)

# Load the trained model
model = joblib.load('xgboost_model.pkl')

def create_time_features(date_string):
    """
    Converts a user-submitted date string into the 6 features
    the model was trained on.
    """
    # Convert string to datetime object
    dt = pd.to_datetime(date_string)
    
    # Create a DataFrame with the exact same column names as your notebook
    data = {
        'hour': [dt.hour],
        'dayofweek': [dt.dayofweek],
        'quarter': [dt.quarter],
        'month': [dt.month],
        'year': [dt.year],
        'dayofyear': [dt.dayofyear]
    }
    
    return pd.DataFrame(data)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # Get date from the form
            date_input = request.form['date_input']
            
            # Preprocess inputs exactly like in the notebook
            input_df = create_time_features(date_input)
            
            # Make prediction
            prediction = model.predict(input_df)
            
            # Format result
            mw_output = round(prediction[0], 2)
            prediction_text = f"Predicted Energy Load: {mw_output} MW"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)