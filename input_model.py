from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Lambda

#First model to get x_i from the input: (x_1,...,x_d)

def model_input(d):
    input = Input(shape=(d,))
    output = {}

    for i in range(1, d + 1):  # Get each node from the input
        output[f'x_{i}'] = Lambda(lambda x: x[:, i - 1:i])(input)

    result = {}    #For each x_i we make one model and put it in a dictionary
    for i in range(1,d+1):
        result[f'x_{i}'] = Model(inputs = input,
                                 outputs = output[f'x_{i}'],
                                 name = f'input_x_{i}')

    return result
    # So model_input = {x_1: Model(input=(x_1,...,x_d), output = x_1), x_2: ....}
    # This way we can split the inputs and compute phi_K(x_q) for q = 1,...,d