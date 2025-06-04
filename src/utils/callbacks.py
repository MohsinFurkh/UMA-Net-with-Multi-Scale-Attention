"""
Custom callbacks for UMA-Net training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class PlotLearning(Callback):
    """
    Callback to plot training metrics after each epoch.
    """
    def __init__(self, metrics=None, output_dir='training_plots'):
        """
        Initialize the callback.
        
        Args:
            metrics: List of metrics to plot. If None, all metrics will be plotted.
            output_dir: Directory to save the plots.
        """
        super().__init__()
        self.metrics = metrics
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def on_train_begin(self, logs=None):
        """Initialize lists for storing metrics."""
        self.metrics_history = {}
        self.val_metrics_history = {}
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Update metrics history
        for metric, value in logs.items():
            if self.metrics is None or any(m in metric for m in self.metrics):
                if 'val_' in metric:
                    if metric not in self.val_metrics_history:
                        self.val_metrics_history[metric] = []
                    self.val_metrics_history[metric].append(float(value))
                else:
                    if metric not in self.metrics_history:
                        self.metrics_history[metric] = []
                    self.metrics_history[metric].append(float(value))
        
        # Plot metrics
        self._plot_metrics(epoch)
    
    def _plot_metrics(self, epoch):
        """Plot training and validation metrics."""
        metrics = [m for m in self.metrics_history.keys() if not m.startswith('val_')]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Plot training metric
            plt.plot(
                np.arange(1, len(self.metrics_history[metric]) + 1),
                self.metrics_history[metric],
                'b-',
                label=f'Training {metric}'
            )
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in self.val_metrics_history:
                plt.plot(
                    np.arange(1, len(self.val_metrics_history[val_metric]) + 1),
                    self.val_metrics_history[val_metric],
                    'r-',
                    label=f'Validation {metric}'
                )
            
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend()
            plt.grid(True)
            
            # Save the figure
            plt.savefig(os.path.join(self.output_dir, f'{metric}_epoch_{epoch + 1}.png'))
            plt.close()


class ModelCheckpointWithBestWeights(tf.keras.callbacks.ModelCheckpoint):
    """
    Extended ModelCheckpoint that saves the model with the best weights
    based on a specific metric.
    """
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq='epoch', **kwargs):
        super().__init__(
            filepath=filepath,
            monitor=monitor,
            verbose=verbose,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            mode=mode,
            save_freq=save_freq,
            **kwargs
        )
        self.best_weights = None
    
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.best_weights = self.model.get_weights()
    
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.monitor_op(current, self.best):
            self.best_weights = self.model.get_weights()
    
    def on_train_end(self, logs=None):
        """Set the model weights to the best weights at the end of training."""
        if self.save_best_only and self.best_weights is not None:
            self.model.set_weights(self.best_weights)


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler with warmup and cosine decay.
    """
    def __init__(self, learning_rate_base, warmup_epochs, epochs, verbose=0):
        """
        Initialize the learning rate scheduler.
        
        Args:
            learning_rate_base: Base learning rate
            warmup_epochs: Number of warmup epochs
            epochs: Total number of training epochs
            verbose: Verbosity mode
        """
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        # Linear warmup followed by cosine decay
        if epoch < self.warmup_epochs:
            lr = self.learning_rate_base * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            lr = 0.5 * self.learning_rate_base * (1 + np.cos(np.pi * progress))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose > 0:
            print(f'\nEpoch {epoch + 1}: Learning rate is {lr:.6f}.')


def get_callbacks(config):
    """
    Get a list of callbacks for model training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    if config.get('checkpoint_dir'):
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        checkpoint_path = os.path.join(
            config['checkpoint_dir'],
            'model_epoch_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
        )
        
        checkpoint = ModelCheckpointWithBestWeights(
            filepath=checkpoint_path,
            monitor=config.get('monitor', 'val_loss'),
            save_best_only=config.get('save_best_only', True),
            save_weights_only=config.get('save_weights_only', False),
            mode=config.get('mode', 'min'),
            verbose=1
        )
        callbacks.append(checkpoint)
    
    # Early stopping
    if config.get('early_stopping'):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=config.get('monitor', 'val_loss'),
            patience=config.get('patience', 10),
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
    
    # Learning rate scheduler
    if config.get('use_lr_scheduler'):
        lr_scheduler = LearningRateScheduler(
            learning_rate_base=config.get('learning_rate', 1e-3),
            warmup_epochs=config.get('warmup_epochs', 5),
            epochs=config.get('epochs', 100),
            verbose=1
        )
        callbacks.append(lr_scheduler)
    
    # TensorBoard
    if config.get('tensorboard_dir'):
        os.makedirs(config['tensorboard_dir'], exist_ok=True)
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=config['tensorboard_dir'],
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
    
    # Plot learning curves
    if config.get('plot_learning_curves'):
        plot_dir = os.path.join(config.get('output_dir', 'output'), 'plots')
        plot_callback = PlotLearning(output_dir=plot_dir)
        callbacks.append(plot_callback)
    
    return callbacks
