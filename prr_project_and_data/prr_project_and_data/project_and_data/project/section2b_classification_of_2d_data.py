import multiprocessing
import numpy as np
import qutip as qt
import valedian211113 as vll
from qutip import *
import random
import math
import matplotlib.pyplot as plt

"""
It is the code used to train the QSNN to classify 2d data.
The result is shown in section II. B.
It is an example of the training and test processes with a specific training set.
"""

"""
Construct quantum stochastic neural networks
"""

N_in = 12
N_out = 2
N = N_in + N_out
H_COMPONENTS = []
for in_neuron_1 in range(0, N_in):
    for in_neuron_2 in range(0, N_in):
        if in_neuron_1 < in_neuron_2:
            H_COMPONENTS.append(qt.basis(N, in_neuron_1) * qt.basis(N, in_neuron_2).dag())


h_num = len(H_COMPONENTS)
C_COMPONENTS = [qt.qzero(N) for h in range(0, h_num)]
for in_neuron in range(0, N_in):
    for out_neuron in range(N_in, N):
        C_COMPONENTS.append(qt.basis(N, out_neuron) * qt.basis(N, in_neuron).dag())


tot_param_num = len(C_COMPONENTS)
for i in range(tot_param_num - h_num):
    H_COMPONENTS.append(qt.qzero(N))


def H(params):
    return sum(
        weight * (H_cm + H_cm.dag())
        for weight, H_cm in zip(params, H_COMPONENTS)
    )


def C_list(params):
     return [
         weight * C_cm
         for weight, C_cm in zip(params, C_COMPONENTS)
         ]

mdl = vll.ChannelModel(H, C_list)

"""
Set the training set
"""

train_set_size = 20  
train_set_all = [[random.uniform(0, 1), random.uniform(0, 1)] for point_train in range(0, 500)]

train_set = []
train_rgb = []
class_1 = 0
class_2 = 0
neuron_class_1 = basis(N, N-2)*basis(N, N-2).dag()  
neuron_class_2 = basis(N, N-1)*basis(N, N-1).dag()  
desired_output_dm_set = []
for point in train_set_all:
    if point[1] > point[0]+0.3 and class_1 < train_set_size / 2:
        train_rgb.append([1, 0, 0])  # The data labeled class 1 is drawn red.
        train_set.append(point)
        desired_output_dm_set.append(neuron_class_1)   # label
        class_1 += 1

    elif point[1] < point[0]-0.3 and class_2 < train_set_size / 2:
        train_set.append(point)
        train_rgb.append([0, 0, 1])  # The data labeled class 1 is drawn blue.
        desired_output_dm_set.append(neuron_class_2)  # label
        class_2 += 1

# the another training set we used
# for point in train_set_all:
#     if point[1] < 0.7-point[0] and class_1 < train_set_size / 2:
#         train_rgb.append([1, 0, 0])
#         train_set.append(point)
#         desired_output_dm_set.append(neuron_class_1)
#         class_1 += 1
#     elif point[1] > 1.3-point[0] and class_2 < train_set_size / 2:
#         train_set.append(point)
#         train_rgb.append([0, 0, 1])
#         desired_output_dm_set.append(neuron_class_2)
#         class_2 += 1

train_set_size = len(train_set)  

"""
Set the test set
"""

x = np.linspace(0.1, 1, 10)
y = np.linspace(0.1, 1, 10)
test_set = []
for i in x:
    for j in y:
        test_set.append([i, j])
test_set_size = len(test_set)
test_desired_output_dm_set = []
for point in test_set:
    if point[1] >= point[0]:
        test_desired_output_dm_set.append(neuron_class_1)
    else:
        test_desired_output_dm_set.append(neuron_class_2)


"""
Initialize the network
"""

# initial value of the Hamiltonian
h_initial = [0.1 for i in range(h_num)]  
gamma_initial = [0.1 for i in range(tot_param_num)]
# all the parameters are time independent
time_inv_list = list(range(len(C_COMPONENTS)))

# evolution time
t_unitary = 1
t_output = 1

# initialize the parameters in (2) Unitary evolution
times_list_unitary = np.linspace(0, t_unitary, 2*t_unitary+1)
parameters_list_unitary = np.zeros((len(times_list_unitary), tot_param_num))
adj_unitary = []
for i in range(0, h_num):
    parameters_list_unitary[:, i] = h_initial[i]
    adj_unitary.append(i)  # the index of parameters that can be updated
ctrl_unitary = vll.Controller(times_list_unitary, parameters_list_unitary, time_invariant=time_inv_list, adjustable=adj_unitary)

# initialize the parameters in (3) Output
times_list_output = np.linspace(0, t_output, 2*t_output+1)
parameters_list_output = np.zeros((len(times_list_output), tot_param_num))
adj_output = []  # the index of parameters that can be updated
for i in range(h_num, tot_param_num):
    parameters_list_output[:, i] = gamma_initial[i]
    adj_output.append(i)
ctrl_output = vll.Controller(times_list_output, parameters_list_output, time_invariant=time_inv_list, adjustable=adj_output)

"""
Encode
convert the classical data into quantum input
"""

input_dm_for_point = []
for point in train_set:
    normalization_coefficient = math.sqrt(sum([point[0] ** (2*i) + point[1] ** (2*i) for i in range(int(N_in/2))]))
    state = sum([point[0] ** i * qt.basis(N, i) + point[1] ** i * qt.basis(N, int(N_in/2)+i) for i in range(int(N_in/2))])/normalization_coefficient    # code 4
    input_dm_for_point.append(state * state.dag())

test_input_dm_for_point = []
for point in test_set:
    normalization_coefficient = math.sqrt(sum([point[0] ** (2*i) + point[1] ** (2*i) for i in range(int(N_in/2))]))
    state = sum([point[0] ** i * qt.basis(N, i) + point[1] ** i * qt.basis(N, int(N_in/2)+i) for i in range(int(N_in/2))])/normalization_coefficient    # code 4
    test_input_dm_for_point.append(state * state.dag())

"""
Define the loss and gradient
"""


def loss_unitary(output_dm, desired_output_dm):
    return -((qt.vector_to_operator(super_operator_output * qt.operator_to_vector(output_dm)) * desired_output_dm).tr())


def dl_df_unitary(output_dm, desired_output_dm, delta_output_dm):
    delta_output_dm_2 = qt.vector_to_operator(super_operator_output * (qt.operator_to_vector(delta_output_dm)))
    return -((delta_output_dm_2 * desired_output_dm).tr()).real


def loss_output(output_dm, desired_output_dm):
    return -((output_dm * desired_output_dm).tr())


def dl_df_output(output_dm, desired_output_dm, delta_output_dm):
    return -((delta_output_dm * desired_output_dm).tr()).real


def pH_pp(params):
    return [H_cm + H_cm.dag() for H_cm in H_COMPONENTS]


def pC_pp_list(params):
    l = [[qt.qzero(N) for i in range(0, tot_param_num)] for j in range(0, tot_param_num)]
    for i in range(0, tot_param_num):
        l[i][i] = C_COMPONENTS[i]
    return l


def unitary_evolution(parameters_list_u, input_dm, desired_output_dm, get_grad):
    """
    evolution (2). unitary evolution
    The gradient of the parameters is evaluated simultaneously after each evolution.
    :return:
    """
    ctrl_unitary.safe_set('parameters', parameters_list_u)
    plc_constant = vll.ParameterizedLindbladChannel(mdl, ctrl_unitary)
    output_dm = plc_constant(input_dm)
    if get_grad:
        sr1 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_unitary, loss=loss_unitary)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_unitary, sr1)
        return vlplc._gradient(input_dm, desired_output_dm), output_dm
    else:
        return output_dm


def output_evolution(parameters_list, input_dm, desired_output_dm, get_grad):
    """
    evolution (3). output
    The gradient of the parameters is evaluated simultaneously after each evolution.
    :return: the
    """
    ctrl_output.safe_set('parameters', parameters_list)
    plc_constant = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    output_dm = plc_constant(input_dm)
    # training
    if get_grad:
        sr2 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_output, loss=loss_output)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_output, sr2)
        grad_o = vlplc._gradient(input_dm, desired_output_dm)
        return grad_o, 1-((output_dm * desired_output_dm).tr())  # gradient, loss
    # test
    else:
        return (output_dm * neuron_class_1).tr(), (output_dm * neuron_class_2).tr(), (output_dm * desired_output_dm).tr()


def get_super_operator_output(param):
    """
    :return: the super operator of the evolution (3) output
    """
    ctrl_output.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    super_2 = plc.get_super_operator()
    return super_2


def tot_solve(p, parameters_list_u, parameters_list_o):
    """
    evolution of the training input and calculation of the gradient
    :param p: the index of the training set
    """
    result_unitary_evolution_list = unitary_evolution(parameters_list_u, input_dm_for_point[p], desired_output_dm_set[p], get_grad=True)
    output_dm_list_unitary = result_unitary_evolution_list[1]
    tot_grad_u = result_unitary_evolution_list[0]
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_unitary, desired_output_dm_set[p], get_grad=True)
    c = result_output_evolution_list[1]
    tot_grad_o = result_output_evolution_list[0]
    return c, tot_grad_u, tot_grad_o


def test_tot_solve(p, parameters_list_u, parameters_list_o):
    """
    evolution of the test input
    :param p: the index of the test set
    """
    test_output_dm_list_unitary = unitary_evolution(parameters_list_u, test_input_dm_for_point[p], test_desired_output_dm_set[p], get_grad=False)
    test_c = output_evolution(parameters_list_o, test_output_dm_list_unitary, test_desired_output_dm_set[p], get_grad=False)
    return test_c


"""
Calculate the loss and gradient in the case of the initial parameters
"""
process_num = 24  # the number of cores used to calculate

super_operator_output = get_super_operator_output(parameters_list_output)
pool = multiprocessing.Pool(processes=process_num)
grad_ave_unitary = np.zeros((len(times_list_unitary), tot_param_num))
cost = 0
grad_ave_output = np.zeros((len(times_list_output), tot_param_num))
result = []
for point in range(0, train_set_size):
    result.append(pool.apply_async(tot_solve, (point, parameters_list_unitary, parameters_list_output, )))
pool.close()
pool.join()
for i in result:
    cost += i.get()[0]/train_set_size
    grad_ave_unitary += i.get()[1]/train_set_size
    grad_ave_output += i.get()[2]/train_set_size

pool = multiprocessing.Pool(processes=process_num)
test_result = []
for point in range(0, test_set_size):
    test_result.append(pool.apply_async(test_tot_solve, (point, parameters_list_unitary, parameters_list_output, )))
pool.close()
pool.join()
accuracy = 0
for i in test_result:
    accuracy += i.get()[2]/test_set_size

"""
Train and test
"""

# define total number of iterations
total_update_num = 5

cost_list = [cost]
accuracy_list = [accuracy]
update_list = [0]
update_num = 0
test_rgb = []
while update_num < total_update_num:
    # define learning rate
    learning_rate_h_0 = 2
    learning_rate_gama_0 = 2

    # update the parameters
    parameters_list_unitary = parameters_list_unitary - learning_rate_h_0 * grad_ave_unitary
    parameters_list_output = parameters_list_output - learning_rate_gama_0 * grad_ave_output

    # calculate the loss and gradient in the case of the new parameters
    super_operator_output = get_super_operator_output(parameters_list_output)
    grad_ave_unitary = np.zeros((len(times_list_unitary), tot_param_num))
    cost = 0
    grad_ave_output = np.zeros((len(times_list_output), tot_param_num))
    pool = multiprocessing.Pool(processes=process_num)
    result = []
    for point in range(0, train_set_size):
        result.append(pool.apply_async(tot_solve, (point, parameters_list_unitary, parameters_list_output,)))
    pool.close()
    pool.join()
    for i in result:
        cost += i.get()[0]/train_set_size
        grad_ave_unitary += i.get()[1]/train_set_size
        grad_ave_output += i.get()[2]/train_set_size

    # use test sets to evaluate the initial model in the case of the new parameters
    pool = multiprocessing.Pool(processes=process_num)
    test_result = []
    for point in range(0, test_set_size):
        test_result.append(
            pool.apply_async(test_tot_solve, (point, parameters_list_unitary, parameters_list_output,)))
    pool.close()
    pool.join()
    accuracy = 0
    for i in test_result:
        accuracy += i.get()[2]/test_set_size

    # get the probabilities of the network judging that the input is the class 1 and class 2
    if update_num == total_update_num - 1:
        for i in test_result:
            test_rgb.append([i.get()[0], 0, i.get()[1]])

    update_num = update_num + 1
    cost_list.append(cost)
    accuracy_list.append(accuracy)
    update_list.append(update_num)

"""
Give the result of the training and test
"""
file_handle = open('classification_of_2d_data_loss.txt', mode='w')
file_handle.write(str(cost_list))
file_handle = open('classification_of_2d_data_accuracy.txt', mode='w')
file_handle.write(str(accuracy_list))
file_handle = open('classification_of_2d_data_train_set.txt', mode='w')
file_handle.write(str(train_set))
file_handle = open('classification_of_2d_data_test_set.txt', mode='w')
file_handle.write(str(test_set))
file_handle = open('classification_of_2d_data_test_rgb.txt', mode='w')
file_handle.write(str(test_rgb))
file_handle = open('classification_of_2d_data_train_rgb.txt', mode='w')
file_handle.write(str(train_rgb))

""" # plot figure
font = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18,
         }
font_in = {'family': 'Times New Roman',
         'weight': 'bold',
         'size':12,
         }
fig = plt.figure(figsize=(8, 9))

dot_size = 25
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter([point[0] for point in train_set], [point[1] for point in train_set], color=train_rgb, s=dot_size)
ax1.set_title("training data", font)
ax1.set_xlabel("x", font)
ax1.set_ylabel("y", font)

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter([point[0] for point in test_set], [point[1] for point in test_set], color=test_rgb, s=dot_size)
ax2.set_title("test data", font)
ax2.set_xlabel("x", font)
ax2.set_ylabel("y", font)
"""





