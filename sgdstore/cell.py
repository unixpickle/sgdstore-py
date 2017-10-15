"""
RNNCell implementations.
"""

import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell # pylint: disable=E0611

from .loss import batched_mse

# pylint: disable=R0902
class Cell(RNNCell):
    """
    A recurrent cell that uses the parameters of a Layer
    as memory.

    Inputs to the cell are projected down to training
    pairs, queries, and hyper-parameters as needed.
    """
    # pylint: disable=R0913
    def __init__(self,
                 layer,
                 train_batch=1,
                 query_batch=1,
                 num_steps=1,
                 init_lr=0.2,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 loss=batched_mse,
                 flatten_output=True,
                 reuse=None):
        """
        Setup a new cell.

        Arguments:
          layer: the Layer for the memory network.
          train_batch: batch size for live SGD training.
          query_batch: batch size for memory queries.
          num_steps: number of SGD steps.
          init_lr: initial SGD step size.
          initializer: initializer for the projection
            matrices.
          loss: batched loss function to minimize.
          flatten_output: flatten the query results.
          reuse: reuse variables in an existing scope.
        """
        super(Cell, self).__init__(_reuse=reuse)
        self._layer = layer
        self._train_batch = train_batch
        self._query_batch = query_batch
        self._num_steps = num_steps
        self._init_lr = init_lr
        self._initializer = initializer
        self._loss = loss
        self._flatten_output = flatten_output

    @property
    def layer(self):
        """
        Get the layer that is used for the cell's memory.
        """
        return self._layer

    @property
    def state_size(self):
        return tuple(map(tf.TensorShape, map(list, self._layer.param_shapes)))

    @property
    def output_size(self):
        if self._flatten_output:
            return int(self._query_batch * np.prod(self._layer.output_shape))
        return tf.TensorShape(list((self._query_batch,) + self._layer.output_shape))

    def random_state(self, batch_size, dtype):
        """
        Generate a batch of a random states.

        Similar to zero_state(), except that the result is
        non-deterministic.
        """
        randoms = zip(*[self._layer.init_params(dtype=dtype)
                        for _ in range(batch_size)])
        return tuple(map(tf.stack, randoms))

    # pylint: disable=W0221
    def call(self, inputs, state):
        new_state = self._train_state(inputs, state)
        outputs = self._run_query(inputs, new_state)
        if self._flatten_output:
            outputs = tf.reshape(outputs, (tf.shape(outputs)[0], self.output_size))
        return outputs, new_state

    def _train_state(self, inputs, state):
        """
        Perform live training on the RNN state to produce
        an updated state.

        Returns:
          A new, trained state tuple.
        """
        in_shape = (self._train_batch,) + self._layer.input_shape
        out_shape = (self._train_batch,) + self._layer.output_shape
        train_ins = self._projection('TrainIn', inputs, in_shape)
        train_targets = self._projection('TrainOut', inputs, out_shape)
        step_sizes = tf.exp(self._projection('LR', inputs, (1,)) +
                            math.log(self._init_lr))
        cur_state = state
        for _ in range(self._num_steps):
            predictions = self._layer.apply(train_ins, cur_state)
            loss = tf.reduce_sum(self._loss(train_targets, predictions))
            cur_state = _gradient_step(cur_state, loss, step_sizes)
        return cur_state

    def _run_query(self, inputs, state):
        """
        Get the result of applying the query.
        """
        in_shape = (self._query_batch,) + self._layer.input_shape
        queries = self._projection('Query', inputs, in_shape)
        return self._layer.apply(queries, list(state))

    def _projection(self, name, inputs, out_shape):
        """
        Project the batch of inputs to a vector of shape
        [batch_size x out_shape].
        """
        total_in_size = int(np.prod(inputs.get_shape().dims[1:]))
        total_out_size = int(np.prod(out_shape))
        weights = tf.get_variable(name,
                                  shape=(total_in_size, total_out_size),
                                  dtype=inputs.dtype,
                                  initializer=self._initializer)

        flat_in = tf.reshape(inputs, (tf.shape(inputs)[0], total_in_size))
        return tf.reshape(tf.matmul(flat_in, weights),
                          (tf.shape(inputs)[0],) + out_shape)

def _gradient_step(state, loss, step_sizes):
    """
    Take a gradient descent step on the state.

    Returns:
      A new state after the step.
    """
    grads = tf.gradients(loss, list(state))
    new_state = []
    for old_state, grad in zip(state, grads):
        bcast_shape = tuple([1] * (len(grad.get_shape()) - 1))
        scale = tf.reshape(step_sizes, (tf.shape(grad)[0],) + bcast_shape)
        new_state.append(old_state + scale*grad)
    return tuple(new_state)
