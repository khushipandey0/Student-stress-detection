"""
Model Training Script
Trains Logistic Regression, SVM, Naive Bayes, and Random Forest models
Implements ensemble learning combining all models
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class StressDetectionModels:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.model_names = ['Logistic Regression', 'SVM', 'Naive Bayes', 'Random Forest']
        
    def train_individual_models(self, X_train, y_train):
        """Train individual models"""
        print("Training individual models...\n")
        
        # Logistic Regression
        print("1. Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['Logistic Regression'] = lr
        print("   ✓ Logistic Regression trained")
        
        # SVM
        print("2. Training SVM...")
        svm = SVC(random_state=42, probability=True)
        svm.fit(X_train, y_train)
        self.models['SVM'] = svm
        print("   ✓ SVM trained")
        
        # Naive Bayes
        print("3. Training Naive Bayes...")
        nb = GaussianNB()
        nb.fit(X_train, y_train)
        self.models['Naive Bayes'] = nb
        print("   ✓ Naive Bayes trained")
        
        # Random Forest
        print("4. Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf
        print("   ✓ Random Forest trained")
        
        print("\nAll individual models trained successfully!")
    
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble model combining all individual models"""
        print("\nTraining Ensemble Model (Voting Classifier)...")
        
        # Create voting classifier with soft voting
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', self.models['Logistic Regression']),
                ('svm', self.models['SVM']),
                ('nb', self.models['Naive Bayes']),
                ('rf', self.models['Random Forest'])
            ],
            voting='soft'
        )
        
        voting_clf.fit(X_train, y_train)
        self.ensemble_model = voting_clf
        
        print("✓ Ensemble model trained successfully!")
    
    def evaluate_models(self, X_test, y_test, label_encoder):
        """Evaluate all models and return results"""
        print("\nEvaluating models...\n")
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': classification_report(
                    y_test, y_pred, 
                    target_names=label_encoder.classes_,
                    output_dict=True
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            print(f"{name}: Accuracy = {accuracy:.4f}")
        
        # Evaluate ensemble model
        y_pred_ensemble = self.ensemble_model.predict(X_test)
        accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
        results['Ensemble'] = {
            'accuracy': accuracy_ensemble,
            'predictions': y_pred_ensemble,
            'classification_report': classification_report(
                y_test, y_pred_ensemble,
                target_names=label_encoder.classes_,
                output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred_ensemble)
        }
        print(f"\nEnsemble Model: Accuracy = {accuracy_ensemble:.4f}")
        
        return results
    
    def save_models(self, save_dir='models'):
        """Save all trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            with open(f'{save_dir}/{filename}', 'wb') as f:
                pickle.dump(model, f)
        
        # Save ensemble model
        with open(f'{save_dir}/ensemble_model.pkl', 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        
        print(f"\nAll models saved to {save_dir}/")

def main():
    """Main training pipeline"""
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("=" * 60)
    print("Student Stress Detection - Model Training")
    print("=" * 60)
    
    # Determine paths
    data_dir = '../data/processed' if os.path.exists('../data/processed') else 'data/processed'
    model_dir = '../models' if os.path.exists('../models') else 'models'
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = pd.read_csv(f'{data_dir}/X_train.csv').values
    X_test = pd.read_csv(f'{data_dir}/X_test.csv').values
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    
    # Load label encoder
    with open(f'{model_dir}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize and train models
    model_trainer = StressDetectionModels()
    
    # Train individual models
    model_trainer.train_individual_models(X_train, y_train)
    
    # Train ensemble model
    model_trainer.train_ensemble_model(X_train, y_train)
    
    # Evaluate models
    results = model_trainer.evaluate_models(X_test, y_test, label_encoder)
    
    # Save models
    model_trainer.save_models(save_dir=model_dir)
    
    # Save evaluation results
    import json
    results_summary = {
        name: {
            'accuracy': float(results[name]['accuracy']),
            'confusion_matrix': results[name]['confusion_matrix'].tolist()
        }
        for name in results.keys()
    }
    
    with open(f'{model_dir}/evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Model training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

