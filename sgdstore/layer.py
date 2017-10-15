"""
Neural networks that can be used as sgdstore blocks.
"""

from abc import ABC, abstractmethod, abstractproperty

import tensorflow as tf

class Layer(ABC):
    """
    A feed-forward twice-differentiable function that can
    be used as an SGDStore block.
    """
    @abstractproperty
    def input_shape(self):
        """
        Get the shape of inputs to the layer.
        """
        pass

    @abstractproperty
    def output_shape(self):
        """
        Get the shape of outputs from the layer.
        """
        pass

    @abstractproperty
    def param_shapes(self):
        """
        Get the shapes of all the network parameters.
        """
        pass

    @abstractmethod
    def init_params(self, dtype=None):
        """
        Generate a list of initial parameter Tensors.
        """
        pass

    @abstractmethod
    def apply(self, input_batches, params_batch):
        """
        Apply a batch of layers to a batch of input
        batches.

        Arguments:
          input_batches: a batch of batches.
          params_batch: a list of parameter batches.
        """
        pass

class Stack(Layer):
    """
    A composition of layers.
    """
    def __init__(self, *layers):
        self.layers = layers
        for layer_1, layer_2 in zip(layers, layers[1:]):
            assert layer_1.output_shape == layer_2.input_shape

    @property
    def input_shape(self):
        return self.layers[0].input_shape

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    @property
    def param_shapes(self):
        return [shape for layer in self.layers for shape in layer.param_shapes]

    def init_params(self, dtype=None):
        return [val for layer in self.layers for val in layer.init_params(dtype=dtype)]

    def apply(self, input_batches, params_batch):
        param_offset = 0
        cur_outs = input_batches
        for layer in self.layers:
            new_offset = param_offset + len(layer.param_shapes)
            sub_batch = params_batch[param_offset : new_offset]
            param_offset = new_offset
            cur_outs = layer.apply(cur_outs, sub_batch)
        return cur_outs

class FC(Layer):
    """
    A fully-connected layer.
    """
    # pylint: disable=R0913
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 activation=tf.nn.tanh,
                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                 bias_initializer=tf.zeros_initializer()):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation or (lambda x: x)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    @property
    def input_shape(self):
        return (self.num_inputs,)

    @property
    def output_shape(self):
        return (self.num_outputs,)

    @property
    def param_shapes(self):
        return [(self.num_inputs, self.num_outputs), (self.num_outputs,)]

    def init_params(self, dtype=None):
        weight_shape, bias_shape = self.param_shapes
        return [self.weights_initializer(weight_shape, dtype=dtype),
                self.bias_initializer(bias_shape, dtype=dtype)]

    def apply(self, input_batches, params_batch):
        assert len(params_batch) == 2
        weight_out = tf.matmul(input_batches, params_batch[0])
        bias_shape = tf.shape(params_batch[1])
        bias_rows = tf.reshape(params_batch[1], (bias_shape[0], 1, bias_shape[1]))
        return self.activation(weight_out + bias_rows)
