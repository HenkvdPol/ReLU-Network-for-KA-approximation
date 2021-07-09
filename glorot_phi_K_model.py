import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import Input, Model
from config import max_weights, f, d, K, beta

Max = max_norm(max_weights(f,d,K,beta))


# Similar as phi_K_model.py but no custom weight initialisations
# For comments on the code, view phi_K_model.py


###################################################################################
################# Making the individual model for each layer ######################

# T_1 = [a_1^x.a_2^x....]_2
def T_1():
    input = Input(shape=(1,))
    x = Dense(1, 'relu',
              kernel_constraint=Max,
              name='T_1_1')(input)
    output = Dense(1, 'relu',
                   kernel_constraint=Max,
                   name='T_1_2')(x)
    return Model(inputs=input,
                 outputs=output,
                 name='T_1')


# S_1 = a_1^x
def S_1():
    input = Input(shape=(1,))
    x = Dense(2, 'relu',
              kernel_constraint=Max,
              name='S_1_1')(input)
    output = Dense(1, 'relu',
                   kernel_constraint=Max,
                   name='S_1_2')(x)
    return Model(inputs=input,
                 outputs=output,
                 name='S_1')

# first result of S_1
def first_result():
    input = Input(shape=(1,))
    output = Dense(1, 'relu',
                   kernel_constraint=Max,
                   name='first_result_Dense')(input)
    return Model(inputs=input,
                 outputs=output)


# [a_j^x . a_j+1^x ... ]_2 = 2*[a_j-1^x . a_j^x ... ]_2 - 2*a_j-1^x
# T_j = 2*T_j-1 - 2*S_j-1
def T_j(j):
    input = Input(shape=(2,))
    x = Dense(1, 'relu',
              kernel_constraint=Max,
              name=f'T_{j}_1')(input)
    x = Dense(1, 'relu',
              kernel_constraint=Max,
              name=f'T_{j}_2')(x)
    output = Dense(1, 'relu',
                   kernel_constraint=Max,
                   name=f'T_{j}_3')(x)
    return Model(inputs=input,
                 outputs=output)


# a_j+1^x = a_1^x * ([a_j^x . a_j+1^x ... ]_2 - a_j^x)
# S_j+1 = S_1(T_j - S_j)
def S_j():
    input = Input(shape=(2,))
    x = Dense(1, 'relu',
              kernel_constraint=Max,
              name='S_j_1')(input)
    output = S_1()(x)
    return Model(inputs=input,
                 outputs=output,
                 name='S_j')


# result = a_j^x * 3 ** -d(j-1) + sum_(n=1)^(j-1) 3** -d(n-1)a_n^x
def result(d, j):
    input = Input(shape=(2,))
    x = Dense(1, 'relu',
              kernel_constraint=Max,
              name=f'result_dense_{d}_{j}_1')(input)
    x = Dense(1, 'relu',
              kernel_constraint=Max,
              name=f'result_dense_{d}_{j}_2')(x)
    output = Dense(1, 'relu',
                   kernel_constraint=Max,
                   name=f'result_dense_{d}_{j}_3')(x)
    return Model(inputs=input,
                 outputs=output,
                 name=f'result_{j}')


###################################################################################
################################### Building the layers ###########################

# First hidden layer
def First_layer():
    input = Input(shape=(1,))
    output = [T_1()(input), S_1()(input)]
    return Model(inputs=input,
                 outputs=output,
                 name='hidden_layer_1')


# Second hidden Layer
def Second_layer():
    input = [Input(shape=(1,)), Input(shape=(1,))]
    output = [T_j(2)(concatenate(input)),
              S_j()(concatenate(input)),
              first_result()(input[1])]
    return Model(inputs=input,
                 outputs=output,
                 name='hidden_layer_2')


# j_th hidden layer for j = 3,...K-1
def j_th_layer(d, K):
    hidden_layer = {}

    for j in range(2, K):
        input = [Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))]
        output = [T_j(j)(concatenate(input[:-1])),
                  S_j()(concatenate(input[:-1])),
                  result(d, j)(concatenate(input[1:]))]
        hidden_layer[f'{j}'] = Model(inputs=input,
                                     outputs=output,
                                     name=f'hidden_layer_{j + 1}')

    return hidden_layer


# output layer
def output_layer(d, K):
    input = [Input(shape=(1,)), Input(shape=(1,)), Input(shape=(1,))]
    output = Dense(1, 'relu',
                   kernel_constraint=Max,
                   name='output_layer_dense_1')(concatenate(input[1:]))
    return Model(inputs=input,
                 outputs=output,
                 name='output_layer')


######################################################################################################
########################### Building the whole model #################################################


def build_glorot_phi_K_model(d, K):
    input = Input(shape=(1,))
    x = First_layer()(input)

    if K == 1:
        output = Dense(1, 'relu',
                       name='K1_dense_1')(x[1])
        return Model(inputs=input,
                     outputs=output,
                     name='K_is_1')

    elif K == 2:
        x = Second_layer()(x)
        output = Dense(1, 'relu',
                       name='K2_dense_1')(concatenate([x[1], x[2]]))
        return Model(inputs=input,
                     outputs=output,
                     name='K_is_2')

    else:  # K > 2
        x = Second_layer()(x)
        for j in range(2, K):
            x = j_th_layer(d, K)[f'{j}'](x)
        output = output_layer(d, K)(x)
        return Model(inputs=input,
                     outputs=output,
                     name=f'K_is_{K}')

    return


######################################################################################################
############## put all the models of phi_K for each i=1,...,d in a dictionary ########################

def glorot_phi_K(d, K):
    result = {}
    for i in range(1, d + 1):
        result[f'x_{i}'] = build_glorot_phi_K_model(d, K)
        result[f'x_{i}']._name = f'phi_K_x_{i}'  # Rename the model

        # Extra for loop to have the weights untrained in the full model
        for lay in result[f'x_{i}'].layers:
            lay.trainable = False

    return result



################# Glorot full model ########################################
from input_model import model_input
from inner_fn_model import inner_function
from outer_fn_model import outer_function

def build_glorot_total_model(d, K):
    input = Input(shape=(d,))
    c = {}
    for i in range(1, d+1):
        c[f'x_{i}'] = model_input(d)[f'x_{i}'](input)
        c[f'x_{i}'] = glorot_phi_K(d, K)[f'x_{i}'](c[f'x_{i}'])

    together = concatenate([c[f'x_{i}'] for i in range(1,d+1)])

    x = inner_function(d)(together)
    output = outer_function(K, d)(x)

    return Model(inputs = input,
                 outputs = output,
                 name ='total_model')
