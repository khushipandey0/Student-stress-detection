"""
Data Generator for Student Stress Detection
Generates synthetic dataset with 10000 rows
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def generate_student_stress_data(n_samples=10000, random_state=42):
    """
    Generate synthetic student stress detection dataset
    
    Features:
    - study_hours: Hours spent studying per day (0-12)
    - sleep_hours: Hours of sleep per night (3-10)
    - exercise_hours: Hours of exercise per week (0-10)
    - social_activities: Number of social activities per week (0-7)
    - assignment_deadlines: Number of upcoming deadlines (0-10)
    - exam_pressure: Exam pressure level (1-10)
    - family_support: Family support level (1-10)
    - financial_stress: Financial stress level (1-10)
    - academic_performance: GPA or academic score (0-100)
    - workload_level: Perceived workload level (1-10)
    
    Target:
    - stress_level: low, medium, high
    """
    np.random.seed(random_state)
    
    # Generate features
    data = {
        'study_hours': np.random.normal(6, 2, n_samples).clip(0, 12),
        'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 10),
        'exercise_hours': np.random.exponential(2, n_samples).clip(0, 10),
        'social_activities': np.random.poisson(3, n_samples).clip(0, 7),
        'assignment_deadlines': np.random.poisson(3, n_samples).clip(0, 10),
        'exam_pressure': np.random.normal(5, 2, n_samples).clip(1, 10),
        'family_support': np.random.normal(7, 2, n_samples).clip(1, 10),
        'financial_stress': np.random.exponential(2, n_samples).clip(1, 10),
        'academic_performance': np.random.normal(75, 15, n_samples).clip(0, 100),
        'workload_level': np.random.normal(6, 2, n_samples).clip(1, 10),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate stress score based on features
    # Higher stress: more study hours, less sleep, more deadlines, high exam pressure, 
    # low family support, high financial stress, low academic performance, high workload
    stress_score = (
        df['study_hours'] * 0.5 +
        (10 - df['sleep_hours']) * 1.5 +
        df['assignment_deadlines'] * 1.2 +
        df['exam_pressure'] * 1.5 +
        (10 - df['family_support']) * 1.0 +
        df['financial_stress'] * 1.3 +
        (100 - df['academic_performance']) * 0.1 +
        df['workload_level'] * 1.2 -
        df['exercise_hours'] * 0.8 -
        df['social_activities'] * 0.5
    )
    
    # Normalize stress score and assign labels
    stress_score = (stress_score - stress_score.min()) / (stress_score.max() - stress_score.min()) * 100
    
    # Add some randomness
    stress_score += np.random.normal(0, 5, n_samples)
    
    # Assign stress levels
    df['stress_level'] = pd.cut(
        stress_score,
        bins=[0, 33, 66, 100],
        labels=['low', 'medium', 'high']
    )
    
    # Convert to string
    df['stress_level'] = df['stress_level'].astype(str)
    
    # Round numerical columns
    numerical_cols = ['study_hours', 'sleep_hours', 'exercise_hours', 'academic_performance']
    for col in numerical_cols:
        df[col] = df[col].round(2)
    
    integer_cols = ['social_activities', 'assignment_deadlines', 'exam_pressure', 
                    'family_support', 'financial_stress', 'workload_level']
    for col in integer_cols:
        df[col] = df[col].round(0).astype(int)
    
    return df

if __name__ == "__main__":
    print("Generating student stress detection dataset...")
    df = generate_student_stress_data(n_samples=10000)
    
    # Save to CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, 'student_stress_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nStress level distribution:")
    print(df['stress_level'].value_counts())
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset saved to: {output_path}")

