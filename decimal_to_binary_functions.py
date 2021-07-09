import numpy as np

def decimal_to_binary(x):
    # converts input x in [0,1] in decimal form into binary form.
    result = ['0.'] # start with 0.
    n = 1
    while x != 0.0:
        if x >= 2**(-n):
            result.append('1')
            x -= 2**(-n)
        else:
            result.append('0')
        n += 1
    return ''.join(result)

def binary_input(input):
    # input is large (n,d)-array. output is the input in binary representation
    result = input.astype(str)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            result[i][j] = decimal_to_binary(input[i][j])
    return result