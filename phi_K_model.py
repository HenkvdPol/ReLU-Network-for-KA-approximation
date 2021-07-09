import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import Input, Model
from config import max_weights, f, d, K, beta

Max = max_norm(max_weights(f,d,K,beta))


##################################################################################
######################### Function to compute phi_K(x) ###########################

def phi(input, d, K):
    # input = [0.a_1^x a_2^x a_3^x ... ]_2 so for example : 0.1010101011
    result = np.zeros(input.shape)

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            x = np.array([int(k) for k in str(input[i][j]).replace('.', '')])
            y = np.sum([2 * x[k] * 3 ** (-d * (k - 1)) for k in range(1, K + 1)])
            result[i][j] = y

    return result  # = sum_j=1^K 2 * a_j^x * 3^(-d(j-1))

######################## Function to compute the inner function #################

def inner_fn(input, d, K):
    # Similar as phi(input, d, K) but works for d dimensions instead of 1
    result = np.zeros(input.shape)

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            x = np.array([int(k) for k in str(input[i][j]).replace('.', '')])
            y = np.sum([2 * x[k] * 3 ** (-d * (k - 1)) for k in range(1, K + 1)])
            result[i][j] = y * 3 ** -(j+1)

    return np.sum(result, axis = 1, keepdims= True) # 3 sum_j=1^K 3^(-j)phi_K(x_j)



###################################################################################
################### Initial weights and biases for the hidden layers ##############

def two_min_two(shape, dtype=None):
    return tf.Variable(
        np.array([[10], [-10]]),
        dtype=dtype,
        name='two_min_two')
    # Because python calculates in
    # decimal numbers we change the weights
    # to 10 in stead of 2. So we can get
    # the next number in the sequence
    # 0.a_1,a_2,a_3,....


def one_min_one(shape, dtype=None):
    return tf.Variable(
        np.array([[1], [-1]]),
        dtype=dtype,
        name='one_min_one')


class K2_output_weight(tf.keras.initializers.Initializer):
    def __init__(self, d):
        self.d = d

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.Variable(
            np.array([[2 * 3 ** -(self.d * (2 - 1))], [2]]),
            dtype=dtype,
            name='K2_output_weight')


class output_weight(tf.keras.initializers.Initializer):
    def __init__(self, d, K):
        self.d = d
        self.K = K

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.Variable(
            np.array([[2 * 3 ** -(self.d * (self.K - 1))], [2]]),
            dtype=dtype,
            name='output_weight')


class result_weight(tf.keras.initializers.Initializer):
    def __init__(self, d, j):
        self.j = j
        self.d = d

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.Variable(
            np.array([[3 ** (-self.d * (self.j - 1))], [1]]),
            dtype=dtype,
            name='result_weight')


class dot_weight(tf.keras.initializers.Initializer):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.Variable(
            np.array([[1 / self.epsilon]]),
            dtype=dtype,
            name='dot_weight')


class dot_bias(tf.keras.initializers.Initializer):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.Variable(
            np.array([-(1 - self.epsilon) / (2 * self.epsilon),
                      -(1 + self.epsilon) / (2 * self.epsilon)]),
            dtype=dtype,
            name='dot_bias')


class constant(tf.keras.initializers.Initializer):
    def __init__(self, value):
        self.value = value

    def __call__(self, shape, dtype=None, **kwargs):
        return tf.Variable(initial_value=tf.fill(dims=shape,
                                                 value=self.value),
                           dtype=dtype,
                           name='constant')


###################################################################################
################# Making the individual model for each layer ######################

# T_1 = [a_1^x.a_2^x....]_2
def T_1():
    input = Input(shape=(1,))
    x = Dense(1, 'relu',
              kernel_initializer=constant(10.),
              kernel_constraint=Max,
              name='T_1_1')(input)
    output = Dense(1, 'relu',
                   kernel_initializer=constant(1.),
                   kernel_constraint=Max,
                   name='T_1_2')(x)
    return Model(inputs=input, # = [0.a_1^x a_2^x ...]_2
                 outputs=output, # = [a_1^x . a_2^x ...]_2
                 name='T_1')


# S_1 = a_1^x
def S_1(epsilon):
    input = Input(shape=(1,))
    x = Dense(2, 'relu',
              kernel_initializer=constant(1 / epsilon),
              bias_initializer=dot_bias(epsilon),
              kernel_constraint=Max,
              name='S_1_1')(input)
    output = Dense(1, 'relu',
                   kernel_initializer=one_min_one,
                   kernel_constraint=Max,
                   name='S_1_2')(x)
    return Model(inputs=input, # = [0.a_1^x a_2^x ...]_2
                 outputs=output, # a_1^x
                 name='S_1')


# first result of S_1
def first_result():
    input = Input(shape=(1,))
    output = Dense(1, 'relu',
                   kernel_initializer=constant(1.),
                   kernel_constraint=Max,
                   name='first_result_Dense')(input)
    return Model(inputs=input,   # a_1^x
                 outputs=output) # a_1^x


# [a_j^x . a_j+1^x ... ]_2 = 2*[a_j-1^x . a_j^x ... ]_2 - 2*a_j-1^x
# T_j = 2*T_j-1 - 2*S_j-1
def T_j(j):
    input = Input(shape=(2,))
    x = Dense(1, 'relu',
              kernel_initializer=two_min_two,
              kernel_constraint=Max,
              name=f'T_{j}_1')(input)
    x = Dense(1, 'relu',
              kernel_initializer=constant(1.),
              kernel_constraint=Max,
              name=f'T_{j}_2')(x)
    output = Dense(1, 'relu',
                   kernel_initializer=constant(1.),
                   kernel_constraint=Max,
                   name=f'T_{j}_3')(x)
    return Model(inputs=input,  # = [a_j-1^x. a_j^x ...]_2 and a_j-1^x
                 outputs=output, # = [a_j^x. a_j+1^x ...]_2
                 name = f'T_{j}')


# a_j+1^x = a_1^x * ([a_j^x . a_j+1^x ... ]_2 - a_j^x)
# S_j+1 = S_1(T_j - S_j)
def S_j(epsilon):
    input = Input(shape=(2,))
    x = Dense(1, 'relu',
              kernel_initializer=one_min_one,
              kernel_constraint=Max,
              name='S_j_1')(input)
    output = S_1(epsilon)(x)
    return Model(inputs=input, # T_j-1 and S_j-1
                 outputs=output, # S_j
                 name='S_j')


# result = a_j^x * 3 ** -d(j-1) + sum_(n=1)^(j-1) 3** -d(n-1)a_n^x
def result(d, j):
    input = Input(shape=(2,))
    x = Dense(1, 'relu',
              kernel_initializer=result_weight(d, j),
              kernel_constraint=Max,
              name=f'result_dense_{d}_{j}_1')(input)
    x = Dense(1, 'relu',
              kernel_initializer=constant(1.),
              kernel_constraint=Max,
              name=f'result_dense_{d}_{j}_2')(x)
    output = Dense(1, 'relu',
                   kernel_initializer=constant(1.),
                   kernel_constraint=Max,
                   name=f'result_dense_{d}_{j}_3')(x)
    return Model(inputs=input, # a_K and sum_(j=1)^(j-1) 3** -d(n-1)a_j^x
                 outputs=output, # phi_K(x)
                 name=f'result_{j}')


###################################################################################
################################### Building the layers ###########################

# First hidden layer
def First_layer(epsilon):
    input = Input(shape=(1,))
    output = [T_1()(input), S_1(epsilon)(input)]
    return Model(inputs=input, # [0.a_1^x a_2^x ...]_2
                 outputs=output, # [a_1^x. a_2^x ...]_2 and a_1^x
                 name='hidden_layer_1')


# Second hidden Layer
def Second_layer(epsilon):
    input = [Input(shape=(1,)), Input(shape=(1,))]
    output = [T_j(2)(concatenate(input)),
              S_j(epsilon)(concatenate(input)),
              first_result()(input[1])]
    return Model(inputs=input, # [a_1^x. a_2^x ...]_2 and a_1^x
                 outputs=output, # [a_2^x. a_3^x ...]_2 and a_2^x and a_1^x
                 name='hidden_layer_2')


# j_th hidden layer for j = 3,...K-1
def j_th_layer(d, K, epsilon):
    hidden_layer = {}

    for j in range(2, K):
        input = [Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))]
        output = [T_j(j)(concatenate(input[:-1])),
                  S_j(epsilon)(concatenate(input[:-1])),
                  result(d, j)(concatenate(input[1:]))]
        hidden_layer[f'{j}'] = Model(inputs=input, # [a_j-1^x. a_j^x ...]_2 and a_j-1^x
                                                   # and sum_(n=1)^(j-1) a_n^x * 3**(-d(j-1))
                                     outputs=output, # [a_j^x. a_j+1^x ...]_2 and a_j^x
                                                     # and sum_(n=1)^j a_n^x * 3**(-d(n-1))
                                     name=f'hidden_layer_{j + 1}')

    return hidden_layer


# output layer
def output_layer(d, K):
    input = [Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))]
    output = Dense(1, 'relu',
                   kernel_initializer=output_weight(d, K),
                   kernel_constraint=Max,
                   name='output_layer_dense_1')(concatenate(input[1:]))
    return Model(inputs=input, # a_K and sum_(n=1)^(K-1) a_n^x * 3**(-d(n-1))
                 outputs=output, # phi_K(x)
                 name='output_layer')


######################################################################################################
########################### Building the whole model #################################################


def build_phi_K_model(d, K, epsilon):
    input = Input(shape=(1,))
    x = First_layer(epsilon)(input)

    if K == 1:
        output = Dense(1, 'relu',
                       kernel_initializer=constant(2.),
                       name='K1_dense_1')(x[1])
        return Model(inputs=input,
                     outputs=output,
                     name='K_is_1')

    elif K == 2:
        x = Second_layer(epsilon)(x)
        output = Dense(1, 'relu',
                       kernel_initializer=K2_output_weight(d),
                       name='K2_dense_1')(concatenate([x[1], x[2]]))
        return Model(inputs=input,
                     outputs=output,
                     name='K_is_2')

    else:  # K > 2
        x = Second_layer(epsilon)(x)
        for j in range(2, K):
            x = j_th_layer(d, K, epsilon)[f'{j}'](x)
        output = output_layer(d, K)(x)
        return Model(inputs=input,
                     outputs=output,
                     name=f'K_is_{K}')

    return


######################################################################################################
############## put all the models of phi_K for each i=1,...,d in a dictionary ########################

def phi_K(d, K, epsilon):
    result = {}
    for i in range(1, d + 1):
        result[f'x_{i}'] = build_phi_K_model(d, K, epsilon)
        result[f'x_{i}']._name = f'phi_K_x_{i}'  # Rename the model

        # Extra for loop to have the weights untrained in the full model
        for lay in result[f'x_{i}'].layers:
            lay.trainable = False

    return result
