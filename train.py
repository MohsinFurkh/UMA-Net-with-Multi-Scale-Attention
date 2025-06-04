"""
Training script for UMA-Net: Adaptive Ensemble Loss and Multi-Scale Attention for Breast Ultrasound Segmentation
"""

import os
import argparse
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from src.models.uma_net import UMA_Net
from src.data.data_loader import load_data
from src.losses.ensemble_loss import EnsembleLoss, DynamicWeightingCallback
from src.utils.metrics import dice_coefficient, iou_coefficient
from src.utils.callbacks import PlotLearning


def parse_args():
    parser = argparse.ArgumentParser(description='Train UMA-Net')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                        help='Path to config file')
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'models'), exist_ok=True)
    os.makedirs(os.path.join(config['output_dir'], 'logs'), exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    train_generator, val_data = load_data(config['data'])
    
    # Initialize model
    print("Initializing model...")
    model = UMA_Net(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        batch_norm=config['model']['batch_norm']
    )
    
    # Define loss functions
    dice_loss = lambda y_true, y_pred: 1 - dice_coefficient(y_true, y_pred)
    bce_loss = BinaryCrossentropy()
    
    # Initialize ensemble loss
    ensemble_loss = EnsembleLoss(
        loss_fns=[bce_loss, dice_loss],
        weights=config['training']['loss_weights']
    )
    
    # Compile model
    optimizer = Adam(learning_rate=config['training']['learning_rate'])
    model.compile(
        optimizer=optimizer,
        loss=ensemble_loss,
        metrics=[dice_coefficient, iou_coefficient, 'accuracy']
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(config['output_dir'], 'models', 'best_model.h5'),
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(config['output_dir'], 'logs'),
            histogram_freq=1
        ),
        DynamicWeightingCallback(
            ensemble_loss=ensemble_loss,
            validation_data=val_data
        ),
        PlotLearning()
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=config['training']['epochs'],
        validation_data=val_data,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(config['output_dir'], 'models', 'final_model.h5'))
    print("Training completed!")


if __name__ == '__main__':
    main()
