# Student Stress Detection System

A comprehensive machine learning project for detecting student stress levels using ensemble learning techniques.

## Project Structure

```
Student_S_detection/
├── data/
│   ├── raw/                    # Raw generated data
│   ├── processed/              # Preprocessed data
│   └── data_generator.py       # Data generation script
├── preprocessing/
│   └── data_preprocessing.py   # Data cleaning and preprocessing
├── models/
│   ├── *.pkl                   # Trained models (generated)
│   └── train_models.py         # Model training script
├── evaluation/
│   ├── *.png                   # Visualization outputs
│   ├── evaluation_report.txt   # Evaluation report
│   └── model_evaluation.py     # Evaluation and visualization
├── web_app/
│   ├── templates/
│   │   ├── index.html          # Main web interface
│   │   └── about.html          # About page
│   └── app.py                  # Flask web application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

- **Data Generation**: Synthetic dataset with 10,000 samples
- **Data Preprocessing**: Cleaning, balancing, and feature scaling
- **Multiple ML Models**: Logistic Regression, SVM, Naive Bayes, Random Forest
- **Ensemble Learning**: Combines all models for improved accuracy
- **Visualizations**: Accuracy comparisons, confusion matrices, data insights
- **Web Application**: User-friendly interface for stress prediction
- **Motivational Suggestions**: Personalized recommendations based on stress level

## Installation

1. Clone the repository:
```bash
cd /home/sahu/Desktop/Student_S_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Data
```bash
python data/data_generator.py
```

### Step 2: Preprocess Data
```bash
python preprocessing/data_preprocessing.py
```

### Step 3: Train Models
```bash
python models/train_models.py
```

### Step 4: Evaluate Models
```bash
python evaluation/model_evaluation.py
```

### Step 5: Run Web Application
```bash
cd web_app
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Model Performance

The system uses an ensemble of four models:
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Random Forest

The ensemble model combines predictions from all models for improved accuracy.

## Web Application Features

- Input form for student data
- Real-time stress level prediction
- Probability distribution for each stress level
- Personalized motivational suggestions
- Beautiful and responsive UI

## Data Features

The model analyzes 10 features:
1. Study hours per day
2. Sleep hours per night
3. Exercise hours per week
4. Social activities per week
5. Assignment deadlines
6. Exam pressure level
7. Family support level
8. Financial stress level
9. Academic performance
10. Workload level

## Output

The system predicts one of three stress levels:
- **Low**: Well-managed stress
- **Medium**: Moderate stress
- **High**: High stress requiring attention

## License

This project is for educational purposes.

## Author

Student Stress Detection System - ML Project

