from tensorflow.keras import Model, Input
from tensorflow.keras.layers import concatenate

#################### importing individual layers ################################

from input_model import model_input
from phi_K_model import phi_K
from inner_fn_model import inner_function
from outer_fn_model import outer_function

##################Setting everything in one model.###############################

def build_total_model(d, K, epsilon):
    input = Input(shape=(d,))
    c = {}
    for i in range(1, d+1):
        c[f'x_{i}'] = model_input(d)[f'x_{i}'](input)
        c[f'x_{i}'] = phi_K(d, K, epsilon)[f'x_{i}'](c[f'x_{i}'])

    together = concatenate([c[f'x_{i}'] for i in range(1,d+1)])

    x = inner_function(d)(together)
    output = outer_function(d, K)(x)

    return Model(inputs = input,
                 outputs = output,
                 name ='total_model')

################# Glorot full model ########################################
from glorot_phi_K_model import glorot_phi_K

def build_glorot_total_model(d, K):
    input = Input(shape=(d,))
    c = {}
    for i in range(1, d+1):
        c[f'x_{i}'] = model_input(d)[f'x_{i}'](input)
        c[f'x_{i}'] = glorot_phi_K(d, K)[f'x_{i}'](c[f'x_{i}'])

    together = concatenate([c[f'x_{i}'] for i in range(1,d+1)])

    x = inner_function(d)(together)
    output = outer_function(d, K)(x)

    return Model(inputs = input,
                 outputs = output,
                 name ='total_model')


################### Total inner function model #############################

def build_total_inner_model(d, K, epsilon):
    input = Input(shape=(d,))
    c = {}
    for i in range(1, d+1):
        c[f'x_{i}'] = model_input(d)[f'x_{i}'](input)
        c[f'x_{i}'] = phi_K(d, K, epsilon)[f'x_{i}'](c[f'x_{i}'])

    together = concatenate([c[f'x_{i}'] for i in range(1,d+1)])

    output = inner_function(d)(together)

    return Model(inputs = input,
                 outputs = output,
                 name ='total_inner_model')