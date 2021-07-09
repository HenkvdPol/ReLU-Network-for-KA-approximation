# this code is there to have no warning from tensorflow
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'

# View results

from config import d, K, f, g, h
from results import check_inner_function_model, \
    check_glorot_uniform,\
    check_multiple_initialisations_glorot_uniform,\
    check_total_loss_with_extra_training,\
    compare_epsilon_in_phi_K,\
    compare_phi_K_initialisations

check_multiple_initialisations_glorot_uniform()



