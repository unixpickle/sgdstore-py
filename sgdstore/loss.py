"""
Batched loss functions.
"""

import tensorflow as tf

def batched_mse(labels, predictions):
    """
    Compute, for each Tensor in the outer dimension, the
    mean-squared error.
    """
    sq_errors = tf.square(labels - predictions)
    indices = list(range(len(labels.get_shape())))[1:]
    return tf.reduce_mean(sq_errors, axis=indices)
