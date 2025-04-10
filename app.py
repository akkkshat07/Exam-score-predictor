import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the trained models and preprocessor
try:
    # Load preprocessor
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load models
    with open('math_model.pkl', 'rb') as f:
        math_model = pickle.load(f)
    
    with open('reading_model.pkl', 'rb') as f:
        reading_model = pickle.load(f)
    
    with open('writing_model.pkl', 'rb') as f:
        writing_model = pickle.load(f)
    
    model_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    model_loaded = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'error': 'Models not loaded correctly. Please check server logs.'
        }), 500
    
    try:
        # Get form data
        data = {
            'Gender': request.form.get('gender'),
            'EthnicGroup': request.form.get('ethnicGroup'),
            'ParentEduc': request.form.get('parentEduc'),
            'LunchType': request.form.get('lunchType'),
            'TestPrep': request.form.get('testPrep'),
            'ParentMaritalStatus': request.form.get('parentMaritalStatus'),
            'PracticeSport': request.form.get('practiceSport'),
            'IsFirstChild': request.form.get('isFirstChild'),
            'NrSiblings': int(request.form.get('nrSiblings')),
            'TransportMeans': request.form.get('transportMeans'),
            'WklyStudyHours': request.form.get('wklyStudyHours')
        }
        
        # Create a DataFrame with user inputs
        input_data = pd.DataFrame(data, index=[0])
        
        # Add engineered features
        input_data['IsParentHighlyEducated'] = input_data['ParentEduc'].apply(
            lambda x: 1 if x in ["bachelor's degree", "master's degree"] else 0
        )
        
        input_data['GoodStudyHabits'] = (
            (input_data['TestPrep'] == 'completed') & 
            (input_data['WklyStudyHours'].isin(['5 - 10', '> 10']))
        ).astype(int)
        
        input_data['StableFamily'] = (input_data['ParentMaritalStatus'] == 'married').astype(int)
        
        input_data['BalancedLifestyle'] = (
            (input_data['PracticeSport'].isin(['regularly', 'sometimes'])) & 
            (input_data['WklyStudyHours'].isin(['5 - 10', '> 10']))
        ).astype(int)
        
        # Process input data
        processed_input = preprocessor.transform(input_data)
        
        # Make predictions
        math_prediction = math_model.predict(processed_input)[0]
        reading_prediction = reading_model.predict(processed_input)[0]
        writing_prediction = writing_model.predict(processed_input)[0]
        
        # Round predictions to nearest integer and ensure they're within valid score range (0-100)
        math_prediction = max(0, min(100, round(math_prediction)))
        reading_prediction = max(0, min(100, round(reading_prediction)))
        writing_prediction = max(0, min(100, round(writing_prediction)))
        
        # Calculate average score
        avg_score = round((math_prediction + reading_prediction + writing_prediction) / 3)
        
        # Determine performance level
        if avg_score >= 90:
            performance = "Excellent"
        elif avg_score >= 80:
            performance = "Very Good"
        elif avg_score >= 70:
            performance = "Good"
        elif avg_score >= 60:
            performance = "Satisfactory"
        elif avg_score >= 50:
            performance = "Needs Improvement"
        else:
            performance = "Unsatisfactory"
        
        # Prepare response
        response = {
            'math_score': int(math_prediction),
            'reading_score': int(reading_prediction),
            'writing_score': int(writing_prediction),
            'avg_score': avg_score,
            'performance': performance
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)