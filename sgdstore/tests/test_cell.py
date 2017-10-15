"""
Test various cell operations.
"""

import unittest

import numpy as np
import tensorflow as tf

from sgdstore.cell import Cell
from sgdstore.layer import Stack, FC

TESTING_BATCH_SIZE = 7

class TestBatches(unittest.TestCase):
    """
    Tests to make sure that batching works.
    """
    def test_equivalence(self):
        """
        Make sure that using a batch is the same thing as
        evaluating each sample separately.
        """
        # pylint: disable=E1129
        with tf.Graph().as_default():
            with tf.Session() as sess:
                cell = _testing_cell()
                input_batch = _testing_inputs(TESTING_BATCH_SIZE)
                state_batch = _testing_state_batch(sess, cell, TESTING_BATCH_SIZE)

                separate_outs, separate_states = _apply_separate(cell, input_batch,
                                                                 state_batch)
                joint_outs, joint_states = _apply_together(cell, input_batch, state_batch)

                sess.run(tf.global_variables_initializer())
                self.assertTrue(_tensors_close(sess, separate_outs, joint_outs))
                for separate_state, joint_state in zip(separate_states, joint_states):
                    self.assertTrue(_tensors_close(sess, separate_state, joint_state))

def _tensors_close(sess, tensor1, tensor2):
    """
    Check if two tensors are close in values.
    """
    out1, out2 = sess.run((tensor1, tensor2))
    arr1, arr2 = np.array(out1), np.array(out2)
    if arr1.shape != arr2.shape:
        return False
    return np.allclose(arr1, arr2, rtol=1e-4, atol=1e-4)

def _testing_cell():
    """
    Produce a cell that can be used for testing.
    """
    fc_kwargs = {'bias_initializer': tf.truncated_normal_initializer()}
    layer = Stack(FC(5, 15, **fc_kwargs), FC(15, 5, **fc_kwargs))
    return Cell(layer, train_batch=3, query_batch=2, num_steps=2)

def _testing_state_batch(sess, cell, num_states):
    """
    Produce a tuple of random states.
    """
    inits = [cell.layer.init_params() for _ in range(num_states)]
    return tuple(map(list, zip(*sess.run(inits))))

def _testing_inputs(num_inputs):
    """
    Produce a dummy batch of inputs for testing a model.
    """
    return np.random.normal(size=(num_inputs, 7, 3))

def _apply_separate(cell, inputs, states):
    """
    Apply a cell to a batch by running it on individual
    samples and concatenating the results.
    """
    separate_outs = []
    separate_states = tuple([] for _ in cell.layer.param_shapes)
    for an_input, a_state in zip(inputs, zip(*states)):
        in_tensor = tf.constant(np.array([an_input]), dtype=tf.float32)
        state_tensors = tuple(tf.constant(np.array([x]), dtype=tf.float32)
                              for x in a_state)
        an_out = cell(in_tensor, state_tensors)
        separate_outs.append(an_out[0][0])
        for i, state in enumerate(an_out[1]):
            separate_states[i].append(state[0])
    return separate_outs, separate_states

def _apply_together(cell, inputs, states):
    """
    Apply a cell to a batch normally.
    """
    return cell(tf.constant(np.array(inputs), dtype=tf.float32),
                tuple(tf.constant(np.array(x), dtype=tf.float32)
                      for x in states))

if __name__ == '__main__':
    unittest.main()
