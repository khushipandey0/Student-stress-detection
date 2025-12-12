"""
Model Evaluation and Visualization Script
Creates accuracy metrics and data insights visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class ModelEvaluator:
    def __init__(self):
        self.results = None
        self.label_encoder = None
        
    def load_results(self, results_path=None):
        """Load evaluation results"""
        import sys
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        if results_path is None:
            results_path = os.path.join(parent_dir, 'models', 'evaluation_results.json')
            if not os.path.exists(results_path):
                results_path = os.path.join('models', 'evaluation_results.json')
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        # Load label encoder
        encoder_path = os.path.join(parent_dir, 'models', 'label_encoder.pkl')
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join('models', 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Results loaded successfully!")
    
    def plot_accuracy_comparison(self, save_path=None):
        """Plot accuracy comparison of all models"""
        import os
        print("Creating accuracy comparison plot...")
        
        if save_path is None:
            eval_dir = '../evaluation' if os.path.exists('../evaluation') else 'evaluation'
            save_path = f'{eval_dir}/accuracy_comparison.png'
        
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylim([0, 1.1])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {save_path}")
    
    def plot_confusion_matrices(self, save_dir=None):
        """Plot confusion matrices for all models"""
        import os
        print("Creating confusion matrices...")
        
        if save_dir is None:
            save_dir = os.path.dirname(os.path.abspath(__file__))
        
        os.makedirs(save_dir, exist_ok=True)
        class_names = self.label_encoder.classes_
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (model_name, model_results) in enumerate(self.results.items()):
            cm = np.array(model_results['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=axes[idx], cbar_kws={'shrink': 0.8})
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {model_results["accuracy"]:.4f}',
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved to {save_dir}/confusion_matrices.png")
    
    def plot_data_insights(self, data_path=None, save_dir=None):
        """Create data insights visualizations"""
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        
        print("Creating data insights visualizations...")
        
        if data_path is None:
            data_path = os.path.join(parent_dir, 'data', 'raw', 'student_stress_data.csv')
            if not os.path.exists(data_path):
                data_path = os.path.join('data', 'raw', 'student_stress_data.csv')
        
        if save_dir is None:
            save_dir = script_dir
        
        df = pd.read_csv(data_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Stress level distribution
        plt.figure(figsize=(8, 6))
        stress_counts = df['stress_level'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        plt.bar(stress_counts.index, stress_counts.values, color=colors)
        plt.xlabel('Stress Level', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Stress Level Distribution', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        for i, v in enumerate(stress_counts.values):
            plt.text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/stress_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Stress distribution plot saved")
        
        # 2. Feature correlation heatmap
        plt.figure(figsize=(12, 10))
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Correlation heatmap saved")
        
        # 3. Feature distributions by stress level
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        key_features = ['study_hours', 'sleep_hours', 'exam_pressure', 
                       'academic_performance', 'workload_level', 'family_support']
        
        for idx, feature in enumerate(key_features):
            for stress_level in ['low', 'medium', 'high']:
                data = df[df['stress_level'] == stress_level][feature]
                axes[idx].hist(data, alpha=0.6, label=stress_level, bins=20)
            
            axes[idx].set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
            axes[idx].set_title(f'{feature.replace("_", " ").title()} by Stress Level', 
                              fontsize=11, fontweight='bold')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Feature distributions plot saved")
        
        # 4. Box plots for key features
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(key_features):
            df.boxplot(column=feature, by='stress_level', ax=axes[idx])
            axes[idx].set_title(f'{feature.replace("_", " ").title()} by Stress Level',
                              fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Stress Level', fontsize=10)
            axes[idx].set_ylabel(feature.replace('_', ' ').title(), fontsize=10)
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/box_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Box plots saved")
    
    def generate_evaluation_report(self, save_path=None):
        """Generate text evaluation report"""
        import os
        print("Generating evaluation report...")
        
        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_path = os.path.join(script_dir, 'evaluation_report.txt')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("STUDENT STRESS DETECTION - MODEL EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("MODEL ACCURACY RESULTS\n")
            f.write("-" * 70 + "\n")
            for model_name, model_results in self.results.items():
                f.write(f"{model_name:30s}: {model_results['accuracy']:.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("BEST MODEL: Ensemble Model\n")
            f.write(f"Best Accuracy: {self.results['Ensemble']['accuracy']:.4f}\n")
            f.write("=" * 70 + "\n")
        
        print(f"✓ Evaluation report saved to {save_path}")

def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("Model Evaluation and Visualization")
    print("=" * 60)
    
    evaluator = ModelEvaluator()
    evaluator.load_results()
    
    # Create visualizations
    evaluator.plot_accuracy_comparison()
    evaluator.plot_confusion_matrices()
    evaluator.plot_data_insights()
    evaluator.generate_evaluation_report()
    
    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

