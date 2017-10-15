"""
Compare the performance of a Vanilla RNN and an sgdstore
model on few-shot polynomial learning.

Models will be required to fit a polynomial ax^2+bx+c with
random a, b, and c.
"""

import numpy as np
import sgdstore
import tensorflow as tf
from tensorflow.contrib.framework import nest # pylint: disable=E0611

BATCH_SIZE = 32
SEQ_LENGTH = 32

def main():
    """
    Setup some models and compare them.
    """
    in_seqs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SEQ_LENGTH, 2))
    out_seqs = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SEQ_LENGTH, 1))
    models = {
        'vanilla': vanilla_model(in_seqs),
        'sgdstore': sgdstore_model(in_seqs)
    }
    losses = {}
    minimizers = {}
    for name, model in models.items():
        losses[name] = tf.losses.mean_squared_error(out_seqs, model)
        adam = tf.train.AdamOptimizer()
        minimizers[name] = adam.minimize(losses[name])
    training_loop(losses, minimizers, in_seqs, out_seqs)

def training_loop(losses, minimizers, in_ph, out_ph):
    """
    Train every algorithm in a loop.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        batch_idx = 0
        while True:
            ins, outs = make_batch()
            feed_dict = {in_ph: ins, out_ph: outs}
            loss, _ = sess.run((losses, minimizers), feed_dict=feed_dict)
            loss_str = ' '.join([key+'='+str(val) for key, val in loss.items()])
            print('batch %d: %s' % (batch_idx, loss_str))
            batch_idx += 1

def vanilla_model(in_seqs):
    """
    Apply a vanilla RNN to the sequences.
    """
    cell = tf.contrib.rnn.BasicRNNCell(128)
    hidden, _ = tf.nn.dynamic_rnn(cell, in_seqs, dtype=tf.float32)
    return tf.contrib.layers.fully_connected(hidden, 1, activation_fn=None)

def sgdstore_model(in_seqs):
    """
    Apply an sgdstore model to the sequences.
    """
    controller = tf.contrib.rnn.BasicRNNCell(64)
    layer = sgdstore.Stack(sgdstore.FC(4, 32), sgdstore.FC(32, 4))
    sgdcell = sgdstore.Cell(layer, train_batch=4, query_batch=4)
    cell = tf.contrib.rnn.MultiRNNCell([controller, sgdcell])

    init_state = (controller.zero_state(1, tf.float32), sgdcell.random_state(1, tf.float32))
    init_vars = [tf.Variable(x) for x in nest.flatten(init_state)]
    repeated_vars = [tf.tile(x, multiples=[BATCH_SIZE]+([1]*(len(x.get_shape())-1)))
                     for x in init_vars]
    init_state = nest.pack_sequence_as(cell.state_size, repeated_vars)

    query_res = tf.nn.dynamic_rnn(cell, in_seqs, initial_state=init_state)[0]
    return tf.contrib.layers.fully_connected(query_res, 1, activation_fn=None)

def make_batch():
    """
    Create a training batch, (in_seqs, out_seqs).
    """
    quad_coeffs = np.random.normal(size=(BATCH_SIZE, 1, 1))
    lin_coeffs = np.random.normal(size=(BATCH_SIZE, 1, 1))
    constants = np.random.normal(size=(BATCH_SIZE, 1, 1))
    x_vals = np.random.normal(size=(BATCH_SIZE, SEQ_LENGTH, 1))
    y_vals = quad_coeffs*np.square(x_vals) + lin_coeffs*x_vals + constants
    y_last = np.zeros_like(y_vals)
    y_last[:, 1:, :] = y_vals[:, :-1, :]
    inputs = np.concatenate((y_last, x_vals), axis=-1)
    return (inputs, y_vals)

if __name__ == '__main__':
    main()
