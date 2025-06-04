"""
Implementation of adaptive ensemble loss for UMA-Net.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

class EnsembleLoss(Loss):
    """
    Adaptive ensemble loss that combines multiple loss functions.
    
    Args:
        loss_fns: List of loss functions to combine
        weights: Initial weights for each loss function
        adaptive: Whether to use adaptive weighting
    """
    def __init__(self, loss_fns, weights=None, adaptive=True, name='ensemble_loss'):
        super().__init__(name=name)
        self.loss_fns = loss_fns
        self.adaptive = adaptive
        
        if weights is None:
            self.weights = [1.0 / len(loss_fns)] * len(loss_fns)
        else:
            self.weights = weights
            
        # Convert weights to trainable variables if adaptive
        if self.adaptive:
            self.weights = [
                tf.Variable(w, dtype=tf.float32, trainable=True) 
                for w in self.weights
            ]
    
    def call(self, y_true, y_pred):
        """
        Compute the ensemble loss.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            
        Returns:
            Weighted sum of individual losses
        """
        total_loss = 0.0
        
        for i, loss_fn in enumerate(self.loss_fns):
            loss = loss_fn(y_true, y_pred)
            
            if self.adaptive:
                # Ensure weights are positive using softplus
                weight = tf.math.softplus(self.weights[i])
                total_loss += weight * loss
            else:
                total_loss += self.weights[i] * loss
        
        return total_loss
    
    def get_weights(self):
        """Get current weights of the ensemble."""
        if self.adaptive:
            return [tf.math.softplus(w).numpy() for w in self.weights]
        return self.weights


class DynamicWeightingCallback(tf.keras.callbacks.Callback):
    """
    Callback to dynamically adjust loss weights during training.
    """
    def __init__(self, ensemble_loss, validation_data, patience=5, factor=0.5, min_delta=1e-4):
        """
        Initialize the callback.
        
        Args:
            ensemble_loss: Instance of EnsembleLoss
            validation_data: Validation data as tuple (x_val, y_val)
            patience: Number of epochs to wait before adjusting weights
            factor: Factor to multiply the learning rate by when reducing
            min_delta: Minimum change in validation loss to qualify as improvement
        """
        super().__init__()
        self.ensemble_loss = ensemble_loss
        self.validation_data = validation_data
        self.patience = patience
        self.factor = factor
        self.min_delta = min_delta
        
        # Initialize variables
        self.wait = 0
        self.best_weights = None
        self.best_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if not self.ensemble_loss.adaptive:
            return
            
        # Get current validation loss
        x_val, y_val = self.validation_data
        val_loss = self.model.evaluate(x_val, y_val, verbose=0)
        
        if isinstance(val_loss, list):
            val_loss = val_loss[0]
        
        # Check if validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            
            # If no improvement for 'patience' epochs, adjust weights
            if self.wait >= self.patience:
                self._adjust_weights()
                self.wait = 0
    
    def _adjust_weights(self):
        """Adjust weights based on current performance."""
        # Get current weights and losses
        weights = self.ensemble_loss.weights
        
        # Compute gradients of each loss component
        with tf.GradientTape() as tape:
            # Compute individual losses on validation set
            x_val, y_val = self.validation_data
            y_pred = self.model(x_val, training=False)
            
            losses = []
            for loss_fn in self.ensemble_loss.loss_fns:
                loss = loss_fn(y_val, y_pred)
                losses.append(tf.reduce_mean(loss))
        
        # Compute gradients of losses w.r.t. weights
        grads = tape.gradient(losses, weights)
        
        # Update weights using gradient descent
        for i, (w, g) in enumerate(zip(weights, grads)):
            if g is not None:
                new_w = w - self.factor * g
                w.assign(new_w)
                
        # Normalize weights to sum to 1
        normalized_weights = tf.nn.softmax([w for w in self.ensemble_loss.weights])
        for i, w in enumerate(self.ensemble_loss.weights):
            w.assign(normalized_weights[i])
        
        print(f"Adjusted loss weights to: {self.ensemble_loss.get_weights()}")
