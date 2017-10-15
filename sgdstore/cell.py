"""
RNNCell implementations.
"""

import math

import numpy as np

# pylint: disable=E0611
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell

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
                 loss=tf.losses.mean_squared_error,
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
        return tf.TensorShape(list((self._query_batch,) + self._layer.output_shape))

    # pylint: disable=W0221
    def call(self, inputs, state):
        new_state = self._train_state(inputs, state)
        return self._run_query(inputs, new_state), new_state

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
            loss = tf.reduce_sum(tf.map_fn(lambda x: self._loss(x[0], x[1]),
                                           (train_targets, predictions),
                                           dtype=tf.float32))
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
