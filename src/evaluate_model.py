import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

def evaluate_model(model_path='models/best_model.keras',
                 class_indices_path='models/class_indices.pkl',
                 output_dir='models/evaluation'):
    """
    Evaluate a trained model on test data and save metrics.
    """
    try:
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        print("Loading model and class indices...")
        model = tf.keras.models.load_model(model_path)
        with open(class_indices_path, 'rb') as f:
            class_indices = pickle.load(f)
        
        
        print("Creating test data generator...")
        from src.data_preprocessing import create_data_generators
        _, _, test_data = create_data_generators()  
        
        
        print("Generating predictions...")
        y_true = test_data.classes
        y_pred = model.predict(test_data, verbose=1).argmax(axis=1)
        
       
        print("Calculating metrics...")
        
       
        report = classification_report(
            y_true, y_pred, 
            target_names=list(class_indices.keys()),
            output_dict=True
        )
        with open(f"{output_dir}/classification_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
       
        with open(f"{output_dir}/test_accuracy.txt", "w") as f:
            f.write(str(report['accuracy']))
        
       
        plt.figure(figsize=(15, 12))
        sns.heatmap(
            confusion_matrix(y_true, y_pred), 
            annot=True, fmt="d",
            xticklabels=class_indices.keys(),
            yticklabels=class_indices.keys(),
            cmap='Blues'
        )
        plt.title("Confusion Matrix", fontsize=16)
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Evaluation completed successfully!")
        print(f"• Model: {model_path}")
        print(f"• Test Accuracy: {report['accuracy']:.4f}")
        print(f"• Reports saved to: {output_dir}")
        
        return report
        
    except Exception as e:
        print(f"✗ Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_model()