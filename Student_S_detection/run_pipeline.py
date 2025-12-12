"""
Main Pipeline Script
Runs the complete student stress detection pipeline
"""

import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in {description}: {e}")
        return False

def main():
    """Run the complete pipeline"""
    print("=" * 70)
    print("STUDENT STRESS DETECTION - COMPLETE PIPELINE")
    print("=" * 70)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    steps = [
        ("data/data_generator.py", "Data Generation (10,000 samples)"),
        ("preprocessing/data_preprocessing.py", "Data Preprocessing and Balancing"),
        ("models/train_models.py", "Model Training (4 models + Ensemble)"),
        ("evaluation/model_evaluation.py", "Model Evaluation and Visualization")
    ]
    
    for script_path, description in steps:
        success = run_script(script_path, description)
        if not success:
            print(f"\nPipeline stopped at: {description}")
            print("Please fix the errors and try again.")
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Check evaluation results in: evaluation/")
    print("2. Start web application: cd web_app && python app.py")
    print("3. Open browser: http://localhost:5000")
    print("=" * 70)

if __name__ == "__main__":
    main()

