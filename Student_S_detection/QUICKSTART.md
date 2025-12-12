# Quick Start Guide

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Complete Pipeline

Run all steps in sequence:
```bash
python run_pipeline.py
```

This will:
1. Generate 10,000 rows of synthetic data
2. Clean and preprocess the data
3. Balance the dataset
4. Train 4 ML models (Logistic Regression, SVM, Naive Bayes, Random Forest)
5. Create an ensemble model
6. Evaluate all models
7. Generate visualizations

## Running Individual Steps

### 1. Generate Data
```bash
cd data
python data_generator.py
```

### 2. Preprocess Data
```bash
cd preprocessing
python data_preprocessing.py
```

### 3. Train Models
```bash
cd models
python train_models.py
```

### 4. Evaluate Models
```bash
cd evaluation
python model_evaluation.py
```

## Running the Web Application

1. Make sure models are trained (run the pipeline first)

2. Start the Flask app:
```bash
cd web_app
python app.py
```

3. Open your browser and go to:
```
http://localhost:5000
```

## Project Structure

- `data/` - Data generation and storage
- `preprocessing/` - Data cleaning and preprocessing
- `models/` - Model training scripts
- `evaluation/` - Model evaluation and visualizations
- `web_app/` - Flask web application

## Output Files

After running the pipeline, you'll find:

- **Data**: `data/raw/student_stress_data.csv`
- **Processed Data**: `data/processed/`
- **Trained Models**: `models/*.pkl`
- **Visualizations**: `evaluation/*.png`
- **Evaluation Report**: `evaluation/evaluation_report.txt`

## Troubleshooting

- If you get import errors, make sure you're running scripts from the project root directory
- If models aren't found, run the training step first
- Make sure all dependencies are installed: `pip install -r requirements.txt`

