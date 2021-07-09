import tensorflow as tf
import numpy as np
from config import d, K, f, g, h
from matplotlib import pyplot as plt
from decimal_to_binary_functions import binary_input

############### Full Model ######################
from full_model import build_total_model, build_total_inner_model

############## phi_K model #######################
from phi_K_model import build_phi_K_model, phi, inner_fn

######## Glorot normal phi_K model ###############
from glorot_phi_K_model import build_glorot_phi_K_model

def check_inner_function_model(K):
    # Check the loss for the inner function for different choices of epsilon and dimension d. K=3
    model1 = {}
    model2 = {}
    model3 = {}
    model4 = {}
    fig, ax = plt.subplots(d, K, sharey=True)

    for i in range(2):
        for j in range(K):
            dataset = np.random.rand(10000, j+1+3*i)
            noise = np.random.normal(0, 0.01, dataset.shape)

            x_train = np.float32(binary_input(dataset[1000:]))
            y_train = inner_fn(binary_input(dataset[1000:]), j+1+3*i, 3) + noise[1000:]

            x_test = np.float32(binary_input(dataset[:1000]))
            y_test = inner_fn(binary_input(dataset[:1000]), j+1+3*i, 3) + noise[:1000]

            model1[f'{i,j}'] = build_total_inner_model(j+1 + 3*i, 3, 10**-1)
            model2[f'{i,j}'] = build_total_inner_model(j+1 + 3*i, 3, 10**-2)
            model3[f'{i,j}'] = build_total_inner_model(j+1 + 3*i, 3, 10**-3)
            model4[f'{i,j}'] = build_total_inner_model(j+1 + 3*i, 3, 10**-4)

            for l in range(len(model1[f'{i,j}'].layers)):
                model1[f'{i, j}'].layers[l].trainable = True
                model2[f'{i, j}'].layers[l].trainable = True
                model3[f'{i, j}'].layers[l].trainable = True
                model4[f'{i, j}'].layers[l].trainable = True

            model1[f'{i, j}'].summary()
            for lay in model1[f'{i,j}'].layers:
                print(lay.get_weights())

            model1[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history1 = model1[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model1[f'{i,j}'].evaluate(x_test, y_test)

            for lay in model1[f'{i,j}'].layers:
                print(lay.get_weights())


            model2[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history2 = model2[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model2[f'{i,j}'].evaluate(x_test, y_test)

            model3[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history3 = model3[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model3[f'{i,j}'].evaluate(x_test, y_test)

            model4[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history4 = model4[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model4[f'{i,j}'].evaluate(x_test, y_test)

            ax[i, j].plot(history1.history['loss'], label=10 ** -1, color= 'blue',  linewidth =0.9)
            ax[i, j].plot(history2.history['loss'], label=10 ** -2, color= 'red',   linewidth =0.9)
            ax[i, j].plot(history3.history['loss'], label=10 ** -3, color= 'green', linewidth =0.9)
            ax[i, j].plot(history4.history['loss'], label=10 ** -4, color='orange', linewidth =0.9)
            ax[i, j].set_title(f'd= {j+1+3*i}')

    for a in ax.flat:
        a.set(xlabel='epochs', ylabel='loss')
        a.label_outer()

    plt.suptitle('K = 3')
    plt.yscale('log')
    plt.legend(title = 'epsilon')
    plt.show()
    return


def check_glorot_uniform():
    # Computes the glorot_phi_K model for 5000 epochs.
    model = build_glorot_phi_K_model(1, 5)

    dataset = np.random.rand(10000, 1)
    noise = np.random.normal(0, 0.01, dataset.shape)

    x_train = np.float32(binary_input(dataset[1000:]))
    y_train = phi(binary_input(dataset[1000:]), 1, 5) + noise[1000:]

    x_test = np.float32(binary_input(dataset[:1000]))
    y_test = phi(binary_input(dataset[:1000]), 1, 5) + noise[:1000]

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
    history = model.fit(x=x_train, y=y_train, epochs=5000, batch_size=256)
    model.evaluate(x=x_test, y=y_test)

    plt.plot(history.history['loss'], label='Glorot Uniform')
    plt.title('loss of glorot uniform initializations with 5000 epochs and learning_rate=0.1, K=5')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.legend(title='initializer')
    plt.show()

    return

def check_multiple_initialisations_glorot_uniform():
    # 50 random initalisations for the glorot uniform initialisations and K = 1, ... , 6
    model = {}
    fig, ax = plt.subplots(2, 3, sharey= True)
    for i in range(50):
        for j1 in range(2):
            for j2 in range(3):
                dataset = np.random.rand(10000, 1)
                noise = np.random.normal(0, 0.01, dataset.shape)

                x_train = np.float32(binary_input(dataset[1000:]))
                y_train = phi(binary_input(dataset[1000:]), 1, j2+1 + j1*3) + noise[1000:]

                x_test = np.float32(binary_input(dataset[:1000]))
                y_test = phi(binary_input(dataset[:1000]), 1, j2+1 + j1*3) + noise[:1000]

                model[f'{i, j1, j2}'] = build_glorot_phi_K_model(1, j2+1 + j1*3)

                model[f'{i, j1, j2}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse',
                                          metrics=['mean_absolute_error'])
                history = model[f'{i, j1, j2}'].fit(x=x_train, y=y_train, epochs=100, batch_size=256, verbose=0)
                model[f'{i, j1 ,j2}'].evaluate(x_test, y_test)

                ax[j1 ,j2].plot(history.history['loss'], label = f'{i+1}')
                ax[j1, j2].set_title(f'K={j2+1+j1*3}')

    for a in ax.flat:
        a.set(xlabel='epochs', ylabel='loss')
        a.label_outer()

    plt.yscale('log')
    plt.show()
    return



def check_total_loss_with_extra_training(f,g,h,d,K):
    # Check total loss for the model for functions f,g,h with d,K
    model1 = {}
    model2 = {}
    model3 = {}
    fig, ax = plt.subplots(d, K, sharey=True)

    for i in range(d):
        for j in range(K):
            dataset = np.random.rand(10000, i+1)
            noise = np.random.normal(0, 0.01, dataset.shape)

            x_train = np.float32(binary_input(dataset[1000:]))
            y_train1 = f(np.float32(binary_input(dataset[1000:]))) + noise[1000:]
            y_train2 = g(np.float32(binary_input(dataset[1000:]))) + noise[1000:]
            y_train3 = h(np.float32(binary_input(dataset[1000:]))) + noise[1000:]

            x_test = np.float32(binary_input(dataset[:1000]))
            y_test1 = f(np.float32(binary_input(dataset[:1000]))) + noise[:1000]
            y_test2 = g(np.float32(binary_input(dataset[:1000]))) + noise[:1000]
            y_test3 = h(np.float32(binary_input(dataset[:1000]))) + noise[:1000]

            print(f'------------------------------------- d={i+1}, K={j+1} -------------------------------------')
            model1[f'{i,j}'] = build_total_model(i+1, j+1, 10**-1)
            model2[f'{i,j}'] = build_total_model(i+1, j+1, 10**-1)
            model3[f'{i,j}'] = build_total_model(i+1, j+1, 10**-1)

            model1[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history11 = model1[f'{i,j}'].fit(x=x_train, y=y_train1, epochs=100, batch_size=256, verbose=0)
            model1[f'{i,j}'].evaluate(x_test, y_test1)

            model2[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history21 = model2[f'{i,j}'].fit(x=x_train, y=y_train2, epochs=100, batch_size=256, verbose=0)
            model2[f'{i,j}'].evaluate(x_test, y_test2)

            model3[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history31 = model3[f'{i,j}'].fit(x=x_train, y=y_train3, epochs=100, batch_size=256, verbose=0)
            model3[f'{i,j}'].evaluate(x_test, y_test3)

            for l in range(len(model1[f'{i,j}'].layers)):
                model1[f'{i, j}'].layers[l].trainable = True
                model2[f'{i, j}'].layers[l].trainable = True
                model3[f'{i, j}'].layers[l].trainable = True

            model1[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history12 = model1[f'{i,j}'].fit(x=x_train, y=y_train1, epochs=200, batch_size=256, verbose=0, initial_epoch=100)
            model1[f'{i,j}'].evaluate(x_test, y_test1)

            model2[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history22 = model2[f'{i,j}'].fit(x=x_train, y=y_train2, epochs=200, batch_size=256, verbose=0, initial_epoch=100)
            model2[f'{i,j}'].evaluate(x_test, y_test2)

            model3[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history32 = model3[f'{i,j}'].fit(x=x_train, y=y_train3, epochs=200, batch_size=256, verbose=0, initial_epoch=100)
            model3[f'{i,j}'].evaluate(x_test, y_test3)

            history1 = history11.history['loss'] + history12.history['loss']
            history2 = history21.history['loss'] + history22.history['loss']
            history3 = history31.history['loss'] + history32.history['loss']

            ax[j,i].plot(history1, label=r'$\Vert\mathbf{x}\Vert_{1}$', color= 'blue',  linewidth =0.9)
            ax[j,i].plot(history2, label=r'$\Vert\mathbf{x}\Vert_{2}$', color= 'red',   linewidth =0.9)
            ax[j,i].plot(history3, label=r'$\Vert\mathbf{x}\Vert_{\infty}$', color= 'green', linewidth =0.9)
            ax[j,i].set_title(f'd={i+1}, K={j+1}')
            ax[j,i].axvline(x= 100, color='grey', linewidth =0.5)

    for a in ax.flat:
        a.set(xlabel='epochs', ylabel='loss')
        a.label_outer()

    plt.yscale('log', basey = 2)
    ax[2,0].legend(title = 'f')
    plt.show()
    return


def compare_phi_K_initialisations(K):
    # Compare different weight initiliaisations for phi_K: glorot uniform and custom
    model1 = {}
    model2 = {}
    fig, ax = plt.subplots(2, K, sharey= True)
    for i in range(2):
        for j in range(K):
            dataset = np.random.rand(10000, 1)
            noise = np.random.normal(0, 0.01, dataset.shape)

            x_train = np.float32(binary_input(dataset[1000:]))
            y_train = phi(binary_input(dataset[1000:]), 1, j+1+ i*K) + noise[1000:]

            x_test = np.float32(binary_input(dataset[:1000]))
            y_test = phi(binary_input(dataset[:1000]), 1, j+1 + i*K) + noise[:1000]

            model1[f'{i,j}'] = build_phi_K_model(1, j+1 + i*K, 10**-1)
            model2[f'{i,j}'] = build_glorot_phi_K_model(1, j+1 + i*K)

            model1[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history1 = model1[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model1[f'{i,j}'].evaluate(x_test, y_test)

            model2[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history2 = model2[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model2[f'{i,j}'].evaluate(x_test, y_test)

            ax[i,j].plot(history1.history['loss'], label = 'KA')
            ax[i,j].plot(history2.history['loss'], label = 'glorot uniform')
            ax[i,j].set_title(f'K = {j+1 + i*K}')

    for a in ax.flat:
        a.set(xlabel='epochs', ylabel='loss')
        a.label_outer()

    plt.yscale('log')
    plt.legend(title = 'initialisation')
    plt.show()

    return


def compare_epsilon_in_phi_K(K):
    # Compare the different choices of epsilon for phi_K
    model1 = {}
    model2 = {}
    model3 = {}
    model4 = {}
    fig, ax = plt.subplots(2, K, sharey=True)

    for i in range(2):
        for j in range(K):
            dataset = np.random.rand(10000, j+1+3*i)
            noise = np.random.normal(0, 0.01, dataset.shape)

            x_train = np.float32(binary_input(dataset[1000:]))
            y_train = phi(binary_input(dataset[1000:]), j+1+3*i, 3) + noise[1000:]

            x_test = np.float32(binary_input(dataset[:1000]))
            y_test = phi(binary_input(dataset[:1000]), j+1+3*i, 3) + noise[:1000]

            print(f'------------------------------------- d={j + 1 + 3*i} -------------------------------------')
            model1[f'{i,j}'] = build_phi_K_model(1, j+1 + 3*i, 10**-1)
            model2[f'{i,j}'] = build_phi_K_model(1, j+1 + 3*i, 10**-2)
            model3[f'{i,j}'] = build_phi_K_model(1, j+1 + 3*i, 10**-3)
            model4[f'{i,j}'] = build_phi_K_model(1, j+1 + 3*i, 10**-4)

            model1[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history1 = model1[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model1[f'{i,j}'].evaluate(x_test, y_test)

            model2[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history2 = model2[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model2[f'{i,j}'].evaluate(x_test, y_test)

            model3[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history3 = model3[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model3[f'{i,j}'].evaluate(x_test, y_test)

            model4[f'{i,j}'].compile(optimizer=tf.keras.optimizers.Adam(), loss='mse', metrics=['mean_absolute_error'])
            history4 = model4[f'{i,j}'].fit(x=x_train, y=y_train, epochs=200, batch_size=256, verbose=0)
            model4[f'{i,j}'].evaluate(x_test, y_test)

            ax[i, j].plot(history1.history['loss'], label=10 ** -1, color= 'blue',  linewidth =0.9)
            ax[i, j].plot(history2.history['loss'], label=10 ** -2, color= 'red',   linewidth =0.9)
            ax[i, j].plot(history3.history['loss'], label=10 ** -3, color= 'green', linewidth =0.9)
            ax[i, j].plot(history4.history['loss'], label=10 ** -4, color='orange', linewidth =0.9)
            ax[i, j].set_title(f'K= {j+1+3*i}')

    for a in ax.flat:
        a.set(xlabel='epochs', ylabel='loss')
        a.label_outer()

    plt.yscale('log')
    plt.legend(title = 'epsilon')
    plt.show()
    return
