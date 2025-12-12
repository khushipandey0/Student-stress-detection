"""
Data Preprocessing Script
Handles data cleaning, balancing, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        print("\nCleaning data...")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values found:\n{missing[missing > 0]}")
            df = df.dropna()
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows...")
            df = df.drop_duplicates()
        
        # Remove outliers using IQR method for numerical columns
        numerical_cols = ['study_hours', 'sleep_hours', 'exercise_hours', 
                         'academic_performance']
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Removing {outliers} outliers from {col}...")
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        print(f"Data shape after cleaning: {df.shape}")
        return df
    
    def balance_data(self, X, y):
        """Balance the dataset using SMOTE"""
        print("\nBalancing data using SMOTE...")
        y_series = pd.Series(y)
        print(f"Original class distribution:\n{y_series.value_counts()}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        print(f"Balanced class distribution:\n{pd.Series(y_balanced).value_counts()}")
        print(f"Balanced data shape: {X_balanced.shape}")
        
        return X_balanced, y_balanced
    
    def prepare_data(self, df):
        """Prepare features and target"""
        print("\nPreparing features and target...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'stress_level']
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['stress_level']
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features: {len(feature_cols)}")
        print(f"Feature columns: {feature_cols}")
        print(f"Target classes: {self.label_encoder.classes_}")
        
        return X, y_encoded, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessor(self, save_dir='../models'):
        """Save preprocessor objects"""
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f'{save_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{save_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(f'{save_dir}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"\nPreprocessor objects saved to {save_dir}")

def main():
    """Main preprocessing pipeline"""
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    preprocessor = DataPreprocessor()
    
    # Load data - adjust path based on where script is run from
    data_path = '../data/raw/student_stress_data.csv'
    if not os.path.exists(data_path):
        data_path = 'data/raw/student_stress_data.csv'
    df = preprocessor.load_data(data_path)
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Prepare features and target
    X, y_encoded, y = preprocessor.prepare_data(df_clean)
    
    # Balance data
    X_balanced, y_balanced = preprocessor.balance_data(X, y_encoded)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_balanced, y_balanced, test_size=0.2
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Save preprocessed data
    processed_dir = '../data/processed'
    if not os.path.exists(processed_dir):
        processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    pd.DataFrame(X_train_scaled, columns=preprocessor.feature_columns).to_csv(
        f'{processed_dir}/X_train.csv', index=False
    )
    pd.DataFrame(X_test_scaled, columns=preprocessor.feature_columns).to_csv(
        f'{processed_dir}/X_test.csv', index=False
    )
    pd.Series(y_train).to_csv(f'{processed_dir}/y_train.csv', index=False)
    pd.Series(y_test).to_csv(f'{processed_dir}/y_test.csv', index=False)
    
    print(f"\nPreprocessed data saved to {processed_dir}/")
    
    # Save preprocessor
    model_dir = '../models'
    if not os.path.exists(model_dir):
        model_dir = 'models'
    preprocessor.save_preprocessor(save_dir=model_dir)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()

