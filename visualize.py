"""
Visualization utilities for UMA-Net.
"""

import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model

from src.models import UMA_Net

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize UMA-Net model and results')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (optional)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    return parser.parse_args()

def visualize_model_architecture(model, output_dir):
    """Visualize the model architecture."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model summary to file
    with open(os.path.join(output_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Plot model architecture
    plot_model(
        model,
        to_file=os.path.join(output_dir, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96
    )

def plot_training_history(history, output_dir):
    """Plot training history."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    if 'loss' in history.history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def visualize_feature_maps(model, input_image, layer_name, output_dir):
    """Visualize feature maps for a specific layer."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a model that will return the outputs of the specified layer
    layer_output = model.get_layer(layer_name).output
    feature_map_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=layer_output
    )
    
    # Get feature maps for the input image
    feature_maps = feature_map_model.predict(tf.expand_dims(input_image, axis=0))
    
    # Plot feature maps
    num_features = min(feature_maps.shape[-1], 16)  # Limit to first 16 features
    plt.figure(figsize=(16, 16))
    
    for i in range(num_features):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f'Feature maps for layer: {layer_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'feature_maps_{layer_name}.png'))
    plt.close()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or create model
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model = tf.keras.models.load_model(args.model_path)
    else:
        print("Creating new model")
        model = UMA_Net(
            input_shape=tuple(config['model']['input_shape']),
            num_classes=config['model']['num_classes'],
            dropout_rate=config['model'].get('dropout_rate', 0.1),
            batch_norm=config['model'].get('batch_norm', True)
        )
    
    # Visualize model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(model, os.path.join(args.output_dir, 'architecture'))
    
    # If history is provided, plot training history
    if hasattr(model, 'history') and model.history is not None:
        print("Plotting training history...")
        plot_training_history(model.history, os.path.join(args.output_dir, 'history'))
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == '__main__':
    main()
