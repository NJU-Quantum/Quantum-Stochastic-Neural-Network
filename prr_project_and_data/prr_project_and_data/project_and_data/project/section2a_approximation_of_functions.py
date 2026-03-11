import numpy as np
import qutip as qt
from qutip import *
import math
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import copy
import valedian211113 as vll
import multiprocessing
from functools import reduce

"""
It is the code used to train the QSNN to approximate functions.
The result is shown in section II. A..
"""

"""
Set the training and test set
"""


def function(func):
    x_train, y_train, x_test, y_test = 0, 0, 0, 0
    if func == "sin":
        x_train = np.linspace(0, 1, 20)
        y_train = (1+np.sin(6*x_train))/6

        x_test = np.linspace(0, 1, 20)
        y_test = (1+np.sin(6*x_test))/6

    elif func == "linear":
        x_train = np.linspace(0.1, 0.9, 11)
        y_train = 0.1 + 0.5*x_train

        x_test = np.linspace(0.1, 0.9, 20)
        y_test = 0.1 + 0.5*x_test

    elif func == "quadratic":
        x_train = np.linspace(0, 0.8, 11)
        y_train = (x_train-0.4)**2

        x_test = np.linspace(0, 0.8, 11)
        y_test = (x_test-0.4)**2
    train_list = [[i, j] for i, j in zip(x_train, y_train)]
    test_list = [[i, j] for i, j in zip(x_test, y_test)]
    return train_list, test_list

train_set, test_set = function("linear")  # an example to approximate a linear function

"""
Construct quantum stochastic neural networks
"""

N_in = 2
N_out = 1
N = sum([N_in, N_out])

H_COMPONENTS = []
for hid_neuron_1 in range(0, N):
    for hid_neuron_2 in range(0, N):
        if hid_neuron_1 < hid_neuron_2:
            H_COMPONENTS.append(qt.basis(N, hid_neuron_1) * qt.basis(N, hid_neuron_2).dag())

h_num = len(H_COMPONENTS)
tot_param_num = h_num


def H(params):
    return sum(
        weight * (H_cm + H_cm.dag())
        for weight, H_cm in zip(params, H_COMPONENTS)
    )


def C_list(params):
    return []

mdl = vll.ChannelModel(H, C_list)


"""
Initialize the network
"""

# initial value of the Hamiltonian
h_initial = [random.uniform(0, 1) for i in range(h_num)]
# all the parameters are time independent
time_inv_list = list(range(tot_param_num))
# evolution time
t_tot = 1
times_list = np.linspace(0, t_tot, 2 * t_tot + 1)
# initialize the parameters in the unitary evolution
parameters_list = np.zeros((len(times_list), tot_param_num))
adj_list = []  
for i in range(0, h_num):
    parameters_list[:, i] = h_initial[i]
    adj_list.append(i)
ctrl = vll.Controller(times_list, parameters_list, time_invariant=time_inv_list, adjustable=adj_list)
neuron_out = qt.basis(N, N - 1) * qt.basis(N, N - 1).dag()


"""
Encode
convert the classical data into quantum input
"""

input_dm_for_point = []
for point in train_set:
    normalization_coefficient = math.sqrt(sum([point[0] ** (2*i) for i in range(N-1)]))
    initial_state = sum([point[0] ** i * qt.basis(N, i) for i in range(N-1)])/normalization_coefficient
    input_dm_for_point.append(initial_state * initial_state.dag())


test_input_dm_for_point = []
for point in test_set:
    normalization_coefficient = math.sqrt(sum([point[0] ** (2*i) for i in range(N-1)]))
    initial_state = sum([point[0] ** i * qt.basis(N, i) for i in range(N-1)])/normalization_coefficient
    test_input_dm_for_point.append(initial_state * initial_state.dag())

desired_output_dm_set = []
for point in train_set:
    desired_output_dm_set.append(point[1])

"""
Define the loss and gradient
"""


def loss(output_dm, desired_output_dm):
    return (desired_output_dm - ((output_dm * neuron_out).tr())) ** 2


def dl_df(output_dm, desired_output_dm, delta_output_dm):
    return -2 * (desired_output_dm - (output_dm * neuron_out).tr()) * (delta_output_dm * neuron_out).tr()


def pH_pp(params):
    return [H_cm + H_cm.dag() for H_cm in H_COMPONENTS]


l = [[qt.qzero(N) for i in range(0, tot_param_num)] for j in range(0, tot_param_num)]


def pC_pp_list(params):
    return len(params) * [[]]


def tot_solve(point_data, param):
    """
    evolution of the training input and calculation of the gradient
    :param point_data: the index of the data in the training set
    :param param: parameters
    :return: gradient, loss
    """
    ctrl.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl)
    sr = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df, loss=loss)
    vlplc = vll.VariationalLearningPLC(mdl, ctrl, sr)
    input_dm = input_dm_for_point[point_data]
    desired_output_dm = desired_output_dm_set[point_data]
    output_dm = plc(input_dm)
    return vlplc._gradient(input_dm, desired_output_dm), (desired_output_dm - ((output_dm * neuron_out).tr()))**2


def test_solve(point_data, param):
    """
    evolution of the test input
    :param point_data: the index of the data in the training set
    :param param: parameters
    :return: the predictioned value of y
    """
    ctrl.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl)
    input_dm = test_input_dm_for_point[point_data]
    output_dm = plc(input_dm)
    return (output_dm * neuron_out).tr()


"""
Calculate the loss and gradient in the case of the initial parameters
"""

pool = multiprocessing.Pool(processes=24)
result = []
for point in range(0, len(train_set)):
    result.append(pool.apply_async(tot_solve, (point, parameters_list, )))
pool.close()
pool.join()
grad_ave = np.zeros((len(times_list), tot_param_num))
cost = 0
for i in result:
    cost += i.get()[1]/len(train_set)
    grad_ave += i.get()[0]/len(train_set)

"""
Use test sets to evaluate the initial model
"""

pool = multiprocessing.Pool(processes=24)
test_result = []
for point in range(0, len(test_set)):
    test_result.append(pool.apply_async(test_solve, (point, parameters_list, )))
pool.close()
pool.join()
untrained_prediction = []
for i in test_result:
    untrained_prediction.append(i.get())
    
"""
Train and test
"""

# define total number of iterations
total_update_num = 2000

update_num = 0
cost_list = [cost]
update_list = [0]
prediction = []
while update_num < total_update_num:
    # define learning rate
    rate_h_0 = 4
    decay = 0.01
    learning_rate_h = rate_h_0 / (1 + decay * update_num)
    # update the parameters
    for t in range(0, len(times_list)):
        for i in range(0, h_num):
            parameters_list[t][i] = parameters_list[t][i] - learning_rate_h * grad_ave[t][i]
            
    # calculate the loss and gradient in the case of the new parameters
    pool = multiprocessing.Pool(processes=24)
    result = []
    for point in range(0, len(train_set)):
        result.append(pool.apply_async(tot_solve, (point, parameters_list, )))
    pool.close()
    pool.join()
    grad_ave = np.zeros((len(times_list), tot_param_num))
    cost = 0
    for i in result:
        cost += i.get()[1]/len(train_set)
        grad_ave += i.get()[0]/len(train_set)
        
    # use test sets to evaluate the initial model after training
    if update_num == total_update_num - 1:
        pool = multiprocessing.Pool(processes=24)
        test_result = []
        for point in range(0, len(test_set)):
            test_result.append(pool.apply_async(test_solve, (point, parameters_list, )))
        pool.close()
        pool.join()
        for i in test_result:
            prediction.append(i.get())

    cost_list.append(cost)
    update_num += 1
    update_list.append(update_num)

"""
Give the result of the training and test
"""

file_handle = open('approximation_of_functions_loss.txt', mode='w')
file_handle.write(str(cost_list))
file_handle = open('approximation_of_functions_prediction.txt', mode='w')
file_handle.write(str(prediction))
file_handle = open('approximation_of_functions_untrained_prediction.txt', mode='w')
file_handle.write(str(untrained_prediction))

""" # plot figure
train_x = [i[0] for i in train_set]
train_y = [i[1] for i in train_set]
test_x = [i[0] for i in test_set]
test_y = [i[1] for i in test_set]

font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_in = {'family': 'Times New Roman',
         'weight': 'bold',
         'size':15,
         }

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(update_list, cost_list, color="blue")
ax1.set_xlabel("iterations")
ax1.set_ylabel("Loss", font)
ax1.semilogy()
plt.tick_params(labelsize=15)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(train_x, train_y, color='blue', s=20, label="train set")
ax2.plot(test_x, prediction, color='red', label="final predict")
ax2.plot(test_x, untrained_prediction, "--", color="green", label='initial predict')
ax2.set_xlabel("x", font)
ax2.set_ylabel("y", font)

plt.tick_params(labelsize=15)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.legend(frameon=False, prop=font_in)
plt.show()
"""