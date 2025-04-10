# Student Exam Score Predictor

A machine learning application that predicts student exam scores based on various demographic and behavioral factors.

## Project Overview

This application uses machine learning models to predict student performance in math, reading, and writing based on factors such as:
- Gender
- Ethnic group
- Parent education level
- Test preparation
- Study habits
- Family structure
- And more

The models were trained on a dataset of student information and performance, achieving prediction accuracy of 70-80% (measured by R² score).

## Features

- Predicts math, reading, and writing scores
- Visualizes predictions with charts
- Provides personalized recommendations based on predicted performance
- Clean, responsive user interface

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM
- **Frontend**: HTML, CSS, JavaScript
- **Visualization**: Chart.js

## How to Use

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open your browser and navigate to `http://localhost:5000`
5. Fill out the student information form and click "Predict Scores"

## Model Information

The application uses ensemble learning techniques to make predictions:
- **Math Score Model**: Uses Ridge regression trained on student features
- **Reading Score Model**: Uses a specialized model optimized for reading performance prediction
- **Writing Score Model**: Employs a regression model to predict writing performance

## Feature Importance

Analysis revealed the most important factors affecting student performance:
1. Parent education level
2. Test preparation 
3. Weekly study hours
4. Lunch type (as a socioeconomic indicator)

## Project Structure

```
├── app.py                        # Flask application
├── exam_score_predictor.ipynb    # Jupyter notebook with model training
├── math_model.pkl                # Trained math score prediction model
├── reading_model.pkl             # Trained reading score prediction model
├── writing_model.pkl             # Trained writing score prediction model
├── preprocessor.pkl              # Data preprocessing pipeline
├── static/                       # Static files
│   └── css/
│       └── style.css             # CSS styling
└── templates/                    # HTML templates
    ├── index.html                # Main page with prediction form
    └── about.html                # Information about the project
```

## Future Improvements

- Add more visualization options
- Implement user accounts for tracking predicted scores over time
- Develop an API for integration with educational systems