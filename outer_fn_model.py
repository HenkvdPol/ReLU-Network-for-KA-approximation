from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import max_norm
from config import max_weights, f, K, d, beta

Max = max_norm(max_weights(f, K, d, beta))

# Model function to compute the outer function,
# which is a shallow ReLU network with width 2 ** (K * d) + 1.
# Weights are random normal initialized.

def outer_function(d, K):
    input = Input(shape=(1,))
    x = Dense(units = 2**(K * d)+1,
              activation= 'relu',
              kernel_initializer='random_normal',
              kernel_constraint= Max)(input)
    output = Dense(units = 1,
                   kernel_initializer='random_normal',
                   kernel_constraint= Max)(x)
    return Model(inputs = input,
                 outputs = output,
                 name = 'outer_function')
