""
Metrics for evaluating segmentation performance.
"""

import numpy as np
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff

def dice_coefficient(y_true, y_pred, smooth=1e-5):
    """
    Dice coefficient metric.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def iou_coefficient(y_true, y_pred, smooth=1e-5):
    """
    Intersection over Union (IoU) metric.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision(y_true, y_pred, smooth=1e-5):
    """
    Precision metric.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Precision score
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    predicted_positives = tf.keras.backend.sum(y_pred_f)
    return (true_positives + smooth) / (predicted_positives + smooth)

def recall(y_true, y_pred, smooth=1e-5):
    """
    Recall metric.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Recall score
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    possible_positives = tf.keras.backend.sum(y_true_f)
    return (true_positives + smooth) / (possible_positives + smooth)

def f1_score(y_true, y_pred, smooth=1e-5):
    """
    F1 score metric.
    
    Args:
        y_true: Ground truth mask
        y_pred: Predicted mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        F1 score
    """
    prec = precision(y_true, y_pred, smooth)
    rec = recall(y_true, y_pred, smooth)
    return 2 * ((prec * rec) / (prec + rec + smooth))

def hausdorff_distance(y_true, y_pred):
    """
    Compute the Hausdorff distance between two binary images.
    
    Args:
        y_true: Ground truth binary mask
        y_pred: Predicted binary mask
        
    Returns:
        Hausdorff distance
    """
    def _get_coordinates(mask):
        # Convert mask to coordinates
        coords = np.argwhere(mask > 0.5)
        if len(coords) == 0:
            # If no foreground pixels, return a point far away
            return np.array([[1e6, 1e6]])
        return coords
    
    try:
        # Convert tensors to numpy arrays if needed
        if tf.is_tensor(y_true):
            y_true = y_true.numpy()
        if tf.is_tensor(y_pred):
            y_pred = y_pred.numpy()
        
        # Ensure binary masks
        y_true = (y_true > 0.5).astype(np.uint8)
        y_pred = (y_pred > 0.5).astype(np.uint8)
        
        # Handle batch dimension
        if len(y_true.shape) > 2:
            distances = []
            for i in range(y_true.shape[0]):
                true_coords = _get_coordinates(y_true[i, ..., 0])
                pred_coords = _get_coordinates(y_pred[i, ..., 0])
                
                # Compute directed Hausdorff distances in both directions
                h1 = directed_hausdorff(true_coords, pred_coords)[0]
                h2 = directed_hausdorff(pred_coords, true_coords)[0]
                
                # Take the maximum of the two directed distances
                distances.append(max(h1, h2))
            
            return np.mean(distances)
        else:
            # Handle single image case
            true_coords = _get_coordinates(y_true[..., 0])
            pred_coords = _get_coordinates(y_pred[..., 0])
            
            h1 = directed_hausdorff(true_coords, pred_coords)[0]
            h2 = directed_hausdorff(pred_coords, true_coords)[0]
            
            return max(h1, h2)
    except Exception as e:
        print(f"Error computing Hausdorff distance: {e}")
        return 0.0
