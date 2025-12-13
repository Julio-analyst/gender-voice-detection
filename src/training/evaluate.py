"""
Model Evaluation Module
Generates metrics, confusion matrix, and reports for academic submission
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
from datetime import datetime
import json

# Handle imports for both module and standalone execution
try:
    from ..utils.config import get_config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.utils.config import get_config


class ModelEvaluator:
    """Evaluates model performance and generates reports"""
    
    def __init__(self, model_type: str = 'lstm'):
        """
        Initialize evaluator
        
        Args:
            model_type: Type of model being evaluated
        """
        self.config = get_config()
        self.model_type = model_type
        self.class_names = ['Laki-laki', 'Perempuan']
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels (0 or 1)
            y_pred: Predicted probabilities (0.0 to 1.0)
            threshold: Classification threshold
            
        Returns:
            Dictionary of metrics
        """
        # Flatten arrays to ensure 1D shape
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
        }
        
        # Add confusion matrix elements
        cm = confusion_matrix(y_true, y_pred_binary)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             threshold: float = 0.5,
                             save_path: Optional[str] = None,
                             show: bool = False) -> str:
        """
        Plot and save confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            save_path: Path to save plot (auto-generated if None)
            show: Whether to display plot
            
        Returns:
            Path to saved plot
        """
        # Flatten arrays to ensure 1D shape
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Convert to binary
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar=True)
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            reports_dir = Path(self.config.get('paths.reports', 'reports'))
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = reports_dir / f'confusion_matrix_{self.model_type}_{timestamp}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None,
                      show: bool = False) -> Tuple[str, float]:
        """
        Plot ROC curve and calculate AUC
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            save_path: Path to save plot
            show: Whether to display plot
            
        Returns:
            Tuple of (plot_path, auc_score)
        """
        # Flatten arrays to ensure 1D shape
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_type.upper()} Model')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            reports_dir = Path(self.config.get('paths.reports', 'reports'))
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = reports_dir / f'roc_curve_{self.model_type}_{timestamp}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 ROC curve saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path), roc_auc
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      threshold: float = 0.5) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Classification report as string
        """
        # Flatten arrays to ensure 1D shape
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        y_pred_binary = (y_pred >= threshold).astype(int)
        report = classification_report(
            y_true, y_pred_binary,
            target_names=self.class_names,
            digits=4
        )
        return report
    
    def save_metrics_to_csv(self, metrics: Dict[str, float],
                           save_path: Optional[str] = None) -> str:
        """
        Save metrics to CSV file for academic submission
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save CSV
            
        Returns:
            Path to saved CSV
        """
        # Create DataFrame
        df = pd.DataFrame([metrics])
        df.insert(0, 'model_type', self.model_type)
        df.insert(1, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Save CSV
        if save_path is None:
            reports_dir = Path(self.config.get('paths.reports', 'reports'))
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = reports_dir / f'metrics_{self.model_type}_{timestamp}.csv'
        
        df.to_csv(save_path, index=False)
        print(f"📄 Metrics saved to CSV: {save_path}")
        
        return str(save_path)
    
    def save_metrics_to_json(self, metrics: Dict[str, float],
                            save_path: Optional[str] = None) -> str:
        """
        Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save JSON
            
        Returns:
            Path to saved JSON
        """
        # Add metadata
        output = {
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Save JSON
        if save_path is None:
            reports_dir = Path(self.config.get('paths.reports', 'reports'))
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = reports_dir / f'metrics_{self.model_type}_{timestamp}.json'
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Metrics saved to JSON: {save_path}")
        
        return str(save_path)
    
    def generate_full_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                           threshold: float = 0.5,
                           save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate complete evaluation report with all visualizations and metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            threshold: Classification threshold
            save_dir: Directory to save all outputs
            
        Returns:
            Dictionary with paths to all generated files
        """
        print(f"\n{'='*80}")
        print(f"Generating Evaluation Report - {self.model_type.upper()} Model")
        print(f"{'='*80}")
        
        # Setup save directory
        if save_dir is None:
            save_dir = Path(self.config.get('paths.reports', 'reports'))
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = save_dir / f'{self.model_type}_{timestamp}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        metrics = self.evaluate(y_true, y_pred, threshold)
        
        # Print metrics
        print("\n" + "="*50)
        print(f"Evaluation Results - {self.model_type.upper()}")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("="*50)
        
        # Generate classification report
        print("\n📋 Classification Report:")
        clf_report = self.generate_classification_report(y_true, y_pred, threshold)
        print(clf_report)
        
        # Save classification report
        report_path = model_dir / 'classification_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Classification Report - {self.model_type.upper()} Model\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(clf_report)
        
        # Plot confusion matrix
        print("\n📊 Generating confusion matrix...")
        cm_path = self.plot_confusion_matrix(
            y_true, y_pred, threshold,
            save_path=model_dir / 'confusion_matrix.png'
        )
        
        # Plot ROC curve
        print("\n📈 Generating ROC curve...")
        roc_path, auc_score = self.plot_roc_curve(
            y_true, y_pred,
            save_path=model_dir / 'roc_curve.png'
        )
        metrics['auc'] = auc_score
        
        # Save metrics
        csv_path = self.save_metrics_to_csv(
            metrics,
            save_path=model_dir / 'metrics.csv'
        )
        
        json_path = self.save_metrics_to_json(
            metrics,
            save_path=model_dir / 'metrics.json'
        )
        
        print(f"\n✅ Evaluation report generated successfully!")
        print(f"📁 Report directory: {model_dir}")
        
        return {
            'directory': str(model_dir),
            'confusion_matrix': cm_path,
            'roc_curve': roc_path,
            'classification_report': str(report_path),
            'metrics_csv': csv_path,
            'metrics_json': json_path,
            'metrics': metrics
        }


# ============================================================================
# TESTING CODE
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Evaluation Module")
    print("=" * 80)
    
    # Generate dummy data
    print("\n📊 Generating dummy test data...")
    np.random.seed(42)
    n_samples = 100
    
    # Simulate predictions (probabilities)
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.rand(n_samples)
    
    # Make predictions somewhat correlated with truth for realistic results
    y_pred = np.where(y_true == 1, 
                     np.clip(y_pred + 0.3, 0, 1),  # Higher prob for class 1
                     np.clip(y_pred - 0.3, 0, 1))  # Lower prob for class 0
    
    print(f"   Samples: {n_samples}")
    print(f"   True distribution: {np.bincount(y_true)}")
    
    # Test evaluator
    print(f"\n{'='*80}")
    print("Testing LSTM Evaluator")
    print(f"{'='*80}")
    
    try:
        evaluator = ModelEvaluator(model_type='lstm')
        
        # Generate full report
        report = evaluator.generate_full_report(y_true, y_pred, threshold=0.5)
        
        print(f"\n✅ Evaluation module working correctly!")
        print(f"\n📁 Generated files:")
        for key, value in report.items():
            if key != 'metrics':
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
