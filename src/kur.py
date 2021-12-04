import tensorflow as tf
from tensorflow.keras.layers import Dense


class DenseKur(Dense):
  """Dense layer with kernel usage regularization.
  """
  def call(self, inputs):
    kernel = self.kernel
    ag = kernel
    ag = ag - tf.reduce_max(ag, axis=-1, keepdims=True)
    ag = tf.nn.softmax(ag)

    outputs = inputs @ (ag * kernel)

    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)

    if self.activation is not None:
      outputs = self.activation(outputs)

    return outputs
