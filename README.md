# ReLU-Network-for-KA-approximation
given beta-smooth function f:[0,1]^d -> R, this deep ReLU network approximates f up to approximation rate of 2^(-beta*K) using 2^Kd parameters. Here, K is a set positive integer and d the dimension.

Here, a function is beta-smooth if there exists constant Q such that

|f(x)-f(y)| <= Q||x-y||^{beta}_{\infty} for all x,y in [0,1]^d.

See (https://arxiv.org/abs/2007.15884) for details of the model. Below I give a short description of the code

## main.py ##
The director file which outputs the results of the model in some graph from results.py. All possible results
are imported.

## results.py ##
file which produces multiple functions to view the results in plot. Note, for the full model,
as K,d gets larger the width of the last hidden layer in the model gets exponential large and therefore
exponentially increases training time.

## config.py ##
To use global variables K and d we use config.py. It also stores beta-smooth functions f,g,h which in this
cases are p-norms for p= 1,2,inf. Moreover it contains the max_weight which is the maximum weight for the parameter
weights in the model (kernel_constraint = max_weight).

## decimal_to_binary_functions.py ##
There are two functions in this file. First, we have decimal_to_binary(x) which for float(x) in decimal
form outputs the float in binary representation. Then binary_input(input) outputs the input in binary representation
for an ndarray.

#### model files ####
To approximate f we use the KA approximation. This means we split the function in an outer function
and inner function. The outer function model is a shallow ReLU network with width 2^(K*d)+1 and
the inner function is a deep ReLU network with 2K hidden layers and width 4. The inner function
model itself consists of phi_K_model.py and inner_fn_model.py. All models are then combined into
one deep ReLU network with network architecture (2K+3, (d,4d,...,4d,2^(Kd)+1,1) ) in full_model.py.
All models/layers have been given names to have better understanding of the model in case of warnings.

## input_model.py ##
First model which from input (x_1, ... ,x_d) returns d models with output x_i in a dictionary
using model_input(d) function. This model gives 0 parameter weights to train.

## phi_K_model.py ##
Second model which has input x_i and outputs phi_K(x_i). It is a 2K hidden layer ReLU network. Function
build_phi_K_model(d, K, epsilon) builds one Deep Network to compute phi_K(x_i) and phi_K(d, K, epsilon)
outputs d Networks and returns it in a dictionary, just as input_model.py. The weights are custom
initialized to compute the desired output. To have it work properly as functions
they are defined as classes inherited from tf.keras.initializers.Initializers.
For function phi_K(d, K, epsilon) all weights are non-trainable initialized. The amount of parameter weights
are dependent on the choice of d and K. For K=1 we have 13d parameters, K=2 has 33d weights and for K>2
we have (24K+33)d parameters. We have irregular results for choice of K as this network is not fully connected.
This file also contains phi(input, d, K) and inner_fn(input, d, K) to compute the real values of the interior
and inner function when checking the Neural Network.

## inner_fn_model.py ##
For input (phi_K(x_1), ... ,phi_K(x_d)) it outputs the inner function sum_j=1 3^(-j)phi_K(x_j)
with inner_function(d) function. The network weights are initialized as untrainable and have weight
[3^-1, 3^-2, ... , 3^-d].  We therefore have d+1 parameter weights in this model.

## outer_fn_model.py ##
Shallow ReLU network with width 2^Kd+1 from one input using outer_function(d, K).
Therefore, it has 3*2^Kd + 4 parameter weights. The weights are random normal initialized.

## full_model.py ##
From this file we have function build_total_model(d, K, epsilon). It is build with
input (x_1, ...,x_d) -> model_input(d) -> phi_K(d, K, epsilon) -> concatenate -> inner_function(d)
-> outer_function(d, K). It also contains build_glorot_total_model(d, K) which is similar to
build_total_model but phi_K is replaced with glorot_phi_K(d, K). Moreover, we have
build_total_inner_model(d, K, epsilon) which is build_total_model up to inner_function from the input.

## glorot_phi_K_model.py ##
Similar as phi_K_model.py with no custom weight initialization. So they are glorot uniform initialized.

## Possible WARNING ##
when the weights of phi_K and glorot_phi_K are set to trainable, the model will give following error when training:
WARNING: Gradients do not exist for variables ....
The reason we have this error is that model T_j for j=K is not connected to the last hidden layer. It computes
a node which is not needed for computing phi_K or glorot_phi_K. Therefore, the Warning can be ignored.

