import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.constraints import max_norm
from config import max_weights, f, d, K, beta

Max = max_norm(max_weights(f,d,K,beta))

# This third model gives sum_j=1^d 3**-q phi_K(x_j), the inner function

class inner_weight(tf.keras.initializers.Initializer):
    def __init__(self, d):
        self.d = d

    def __call__(self, shape, dtype=None):
        return tf.Variable(
            np.array([[3 ** -q] for q in range(1, self.d + 1)]),
            dtype=dtype,
            name='inner_weight')


def inner_function(d):
    input = Input(shape=(d,))
    output = Dense(units=1,
                   kernel_initializer=inner_weight(d),
                   kernel_constraint=Max)(input)
    result = Model(inputs=input,
                   outputs=output,
                   name='inner_function')
    result.layers[1].trainable = False # Have the layer untrainable
    return result
