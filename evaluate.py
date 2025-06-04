"""
Evaluation script for UMA-Net model.
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)

from src.models import UMA_Net
from src.data import load_data
from src.utils.metrics import (
    dice_coefficient,
    hausdorff_distance
)

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate UMA-Net model')
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model')
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    return parser.parse_args()

def load_model(model_path):
    """Load a saved model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Check if it's a full model or just weights
    if model_path.endswith('.h5') or os.path.isdir(model_path):
        # Load full model
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'dice_coefficient': dice_coefficient,
                    'EnsembleLoss': None,  # Will be loaded from config
                }
            )
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to load as weights...")
            
            # Try loading weights
            model = UMA_Net(input_shape=(256, 256, 3))  # Default shape, will be overridden
            model.load_weights(model_path)
            return model
    else:
        raise ValueError("Unsupported model format. Please provide a .h5 file or a saved model directory.")

def evaluate_model(model, test_data, config):
    """Evaluate model on test data."""
    X_test, y_test = test_data
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test, batch_size=config.get('batch_size', 8))
    
    # Threshold predictions
    y_pred_bin = (y_pred > 0.5).astype(np.float32)
    
    # Flatten for sklearn metrics
    y_test_flat = y_test.reshape(-1)
    y_pred_flat = y_pred_bin.reshape(-1)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {
        'accuracy': accuracy_score(y_test_flat, y_pred_flat),
        'precision': precision_score(y_test_flat, y_pred_flat, average='weighted'),
        'recall': recall_score(y_test_flat, y_pred_flat, average='weighted'),
        'f1': f1_score(y_test_flat, y_pred_flat, average='weighted'),
        'iou': jaccard_score(y_test_flat, y_pred_flat, average='weighted'),
    }
    
    # Calculate dice coefficient
    dice = 0.0
    for i in range(len(X_test)):
        dice += dice_coefficient(
            tf.convert_to_tensor(y_test[i]),
            tf.convert_to_tensor(y_pred[i])
        )
    metrics['dice'] = dice / len(X_test)
    
    # Calculate Hausdorff distance
    hausdorff = 0.0
    for i in range(len(X_test)):
        hausdorff += hausdorff_distance(
            y_test[i],
            y_pred[i]
        )
    metrics['hausdorff'] = hausdorff / len(X_test)
    
    # Calculate ROC AUC if needed
    if len(np.unique(y_test_flat)) > 1:  # Check if we have both classes
        try:
            metrics['roc_auc'] = roc_auc_score(y_test_flat, y_pred.reshape(-1))
            metrics['pr_auc'] = average_precision_score(y_test_flat, y_pred.reshape(-1))
        except Exception as e:
            print(f"Could not calculate AUC metrics: {e}")
    
    return metrics, y_pred

def plot_results(images, true_masks, pred_masks, output_dir, num_samples=5):
    """Plot sample predictions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original image
        axes[0].imshow(images[idx])
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot ground truth
        axes[1].imshow(true_masks[idx, :, :, 0], cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Plot prediction
        axes[2].imshow(pred_masks[idx, :, :, 0] > 0.5, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'))
        plt.close()

def save_metrics(metrics, output_dir):
    """Save metrics to a file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to text file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{name}: {value:.4f}\n")
            else:
                f.write(f"{name}: {value}\n")
    
    # Save metrics to JSON
    import json
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        # Convert numpy types to Python native types
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                metrics_serializable[k] = float(v)
            else:
                metrics_serializable[k] = v
        json.dump(metrics_serializable, f, indent=2)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    config['test_data_dir'] = args.test_data_dir
    config['output_dir'] = args.output_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")
    test_data = load_data({
        'data_dir': args.test_data_dir,
        'batch_size': config.get('batch_size', 8),
        'image_size': config.get('image_size', (256, 256)),
        'val_split': 0.0  # No validation split for test data
    })
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    
    # Evaluate model
    print("Evaluating model...")
    metrics, y_pred = evaluate_model(model, test_data, config)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print("-" * 30)
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    save_metrics(metrics, args.output_dir)
    
    # Plot sample results
    X_test, y_test = test_data
    plot_results(X_test, y_test, y_pred, os.path.join(args.output_dir, 'samples'))
    
    print("Evaluation completed!")

if __name__ == '__main__':
    main()
