# Student Stress Detection Project - Summary

## Project Overview

A complete machine learning project for detecting student stress levels using ensemble learning techniques.

## Features Implemented

### ✅ 1. Data Generation
- **File**: `data/data_generator.py`
- Generates 10,000 synthetic student records
- 10 features: study_hours, sleep_hours, exercise_hours, social_activities, assignment_deadlines, exam_pressure, family_support, financial_stress, academic_performance, workload_level
- Target variable: stress_level (low, medium, high)

### ✅ 2. Data Preprocessing
- **File**: `preprocessing/data_preprocessing.py`
- Data cleaning (missing values, duplicates, outliers)
- Data balancing using SMOTE
- Feature scaling using StandardScaler
- Train-test split (80-20)
- Saves preprocessed data and preprocessor objects

### ✅ 3. Model Training
- **File**: `models/train_models.py`
- Trains 4 individual models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Random Forest
- Creates ensemble model using Voting Classifier
- Saves all trained models

### ✅ 4. Model Evaluation
- **File**: `evaluation/model_evaluation.py`
- Calculates accuracy for all models
- Generates visualizations:
  - Accuracy comparison bar chart
  - Confusion matrices for all models
  - Stress level distribution
  - Feature correlation heatmap
  - Feature distributions by stress level
  - Box plots for key features
- Creates evaluation report

### ✅ 5. Web Application
- **File**: `web_app/app.py`
- Flask-based web interface
- Input form for student data
- Real-time stress level prediction
- Probability distribution display
- Personalized motivational suggestions
- Beautiful, responsive UI

### ✅ 6. Project Structure
```
Student_S_detection/
├── data/
│   ├── raw/                    # Raw generated data
│   ├── processed/              # Preprocessed data
│   └── data_generator.py
├── preprocessing/
│   └── data_preprocessing.py
├── models/
│   └── train_models.py
├── evaluation/
│   └── model_evaluation.py
├── web_app/
│   ├── templates/
│   │   ├── index.html
│   │   └── about.html
│   └── app.py
├── requirements.txt
├── README.md
├── QUICKSTART.md
└── run_pipeline.py
```

## How to Use

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run complete pipeline**:
   ```bash
   python run_pipeline.py
   ```

3. **Start web application**:
   ```bash
   cd web_app
   python app.py
   ```

4. **Access web interface**:
   Open browser: `http://localhost:5000`

## Model Performance

The ensemble model combines predictions from all 4 models for improved accuracy. Individual model accuracies and ensemble accuracy are displayed in the evaluation visualizations.

## Key Technologies

- **Python 3.x**
- **scikit-learn**: Machine learning models
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib/seaborn**: Visualizations
- **Flask**: Web framework
- **SMOTE**: Data balancing
- **imbalanced-learn**: Handling imbalanced data

## Output Files

- Generated data: `data/raw/student_stress_data.csv`
- Trained models: `models/*.pkl`
- Visualizations: `evaluation/*.png`
- Evaluation report: `evaluation/evaluation_report.txt`

## Next Steps

1. Run the pipeline to generate data and train models
2. Check evaluation results and visualizations
3. Start the web application
4. Test with new student data through the web interface

