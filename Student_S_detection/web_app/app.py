"""
Flask Web Application for Student Stress Detection
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models and preprocessors
# Determine paths based on where script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

MODEL_DIR = os.path.join(parent_dir, 'models')
if not os.path.exists(MODEL_DIR):
    MODEL_DIR = 'models'

DATA_DIR = os.path.join(parent_dir, 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = 'data'

# Load ensemble model
with open(os.path.join(MODEL_DIR, 'ensemble_model.pkl'), 'rb') as f:
    ensemble_model = pickle.load(f)

# Load preprocessors
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'feature_columns.pkl'), 'rb') as f:
    feature_columns = pickle.load(f)

# Motivational suggestions based on stress level
MOTIVATIONAL_SUGGESTIONS = {
    'low': [
        "Great job maintaining a healthy balance! Keep up the excellent work.",
        "You're managing your stress well. Continue with your current routine.",
        "Your stress levels are low - you're on the right track!",
        "Maintain your healthy habits and continue prioritizing self-care.",
        "Excellent stress management! Keep balancing work and relaxation."
    ],
    'medium': [
        "You're doing well, but there's room for improvement. Try to get more sleep.",
        "Consider adding some exercise to your routine - it helps reduce stress.",
        "Take regular breaks during study sessions to prevent burnout.",
        "Try to reduce your workload by prioritizing the most important tasks.",
        "Spend more time with friends and family - social support is important.",
        "Practice mindfulness or meditation for 10 minutes daily.",
        "Make sure you're getting enough sleep - aim for 7-9 hours per night."
    ],
    'high': [
        "Your stress levels are high. It's important to take care of yourself.",
        "Please consider speaking with a counselor or mental health professional.",
        "Prioritize sleep - aim for at least 7-8 hours per night.",
        "Take breaks and don't overwork yourself. Your health comes first.",
        "Try to reduce your study hours and focus on quality over quantity.",
        "Engage in physical activity - even a 20-minute walk can help.",
        "Reach out to family and friends for support - you don't have to do this alone.",
        "Consider talking to your professors about deadline extensions if needed.",
        "Practice deep breathing exercises when you feel overwhelmed.",
        "Remember: It's okay to ask for help. Your well-being matters most."
    ]
}

def get_motivational_suggestion(stress_level):
    """Get a random motivational suggestion based on stress level"""
    import random
    suggestions = MOTIVATIONAL_SUGGESTIONS.get(stress_level.lower(), 
                                               ["Take care of yourself!"])
    return random.choice(suggestions)

def predict_stress_level(features_dict):
    """Predict stress level from input features"""
    # Convert to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Ensure correct column order
    df = df[feature_columns]
    
    # Scale features
    features_scaled = scaler.transform(df)
    
    # Predict
    prediction_encoded = ensemble_model.predict(features_scaled)[0]
    prediction_proba = ensemble_model.predict_proba(features_scaled)[0]
    
    # Decode prediction
    stress_level = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get probabilities for each class
    probabilities = {
        label_encoder.classes_[i]: float(prob) 
        for i, prob in enumerate(prediction_proba)
    }
    
    return stress_level, probabilities

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get input data
        data = request.get_json()
        
        # Extract features
        features = {
            'study_hours': float(data.get('study_hours', 0)),
            'sleep_hours': float(data.get('sleep_hours', 0)),
            'exercise_hours': float(data.get('exercise_hours', 0)),
            'social_activities': int(data.get('social_activities', 0)),
            'assignment_deadlines': int(data.get('assignment_deadlines', 0)),
            'exam_pressure': int(data.get('exam_pressure', 0)),
            'family_support': int(data.get('family_support', 0)),
            'financial_stress': int(data.get('financial_stress', 0)),
            'academic_performance': float(data.get('academic_performance', 0)),
            'workload_level': int(data.get('workload_level', 0))
        }
        
        # Predict
        stress_level, probabilities = predict_stress_level(features)
        
        # Get motivational suggestion
        suggestion = get_motivational_suggestion(stress_level)
        
        # Prepare response
        response = {
            'success': True,
            'stress_level': stress_level,
            'probabilities': probabilities,
            'suggestion': suggestion,
            'features': features
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

