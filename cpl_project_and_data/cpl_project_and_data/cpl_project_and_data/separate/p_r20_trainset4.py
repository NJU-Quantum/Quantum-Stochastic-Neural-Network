import numpy as np
import qutip as qt
from qutip import *
import math
import cmath
import random
import matplotlib.pyplot as plt
import valedian0910 as vll
import multiprocessing


# 模型
in_layer = 4
hidden_layer = 4
out_layer = 2
N = sum([in_layer, hidden_layer, out_layer])
H_COMPONENTS = []
for in_neuron_1 in range(0, in_layer):  # 输入层之间
    for in_neuron_2 in range(0, in_layer):
        if in_neuron_1 < in_neuron_2:
            H_COMPONENTS.append(qt.basis(N, in_neuron_1) * qt.basis(N, in_neuron_2).dag())


for in_neuron in range(0, in_layer):   # 输入层到隐藏层
    for hid1_neuron in range(in_layer, in_layer+hidden_layer):
        if in_neuron < hid1_neuron:
            H_COMPONENTS.append(qt.basis(N, hid1_neuron) * qt.basis(N, in_neuron).dag())
# print(H_COMPONENTS)
h_num = len(H_COMPONENTS)

C_COMPONENTS = [qt.qzero(N) for h in range(0, h_num)]
for in_neuron in range(0, in_layer):    # 输入层到隐藏层
    for hid1_neuron in range(in_layer, in_layer+hidden_layer):
        C_COMPONENTS.append(qt.basis(N, hid1_neuron) * qt.basis(N, in_neuron).dag())

for hid1_neuron in range(in_layer, in_layer+hidden_layer):   # 隐藏层到输出层
    for out_neuron in range(in_layer+hidden_layer, N):
        C_COMPONENTS.append(qt.basis(N, out_neuron) * qt.basis(N, hid1_neuron).dag())
tot_param_num = len(C_COMPONENTS)
# print(C_COMPONENTS)
c_num = tot_param_num - h_num
for i in range(c_num):
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

t_tot = 10
time_inv_list = []   # 所有参数时间不变
for i in range(0, len(C_COMPONENTS)):
    time_inv_list.append(i)

h_initial = [random.uniform(0, 2) for i in range(h_num)]  # 哈密顿量连接#
# h_initial = [0.6657279197829646, 0.4340984507533857, 0.9902767822529691]
print(h_initial)
gama_initial = [0.5 for i in range(tot_param_num)]

times_list = np.linspace(0, t_tot, 2*t_tot+1)
parameters_list = np.zeros((len(times_list), tot_param_num))
adj_list = []  # 哈密顿量和输出耗散参数可被调整
for i in range(0, h_num):
    parameters_list[:, i] = h_initial[i]
    adj_list.append(i)
for i in range(h_num, tot_param_num):
    parameters_list[:, i] = gama_initial[i]
    adj_list.append(i)


def seperate(p, theta, phi):
    phi = (math.cos(theta/2)*basis(N, 1) + cmath.exp(1j*phi)*math.sin(theta/2)*basis(N, 2))
    rho = p * phi * phi.dag() + (1-p) * sum([basis(N, i)*basis(N, i).dag() for i in range(4)])/4
    return rho

measure_0 = basis(N, N-2)*basis(N, N-2).dag()
measure_1 = basis(N, N-1)*basis(N, N-1).dag()
desired_output_dm_set = []
input_dm_set = []

for p in np.arange(0, 1/3, 0.2):   # 分离态， 包含1/3
    input_dm_set.append(seperate(p, math.pi/2, math.pi))
    desired_output_dm_set.append(measure_0)
for p in np.arange(0.4, 1, 0.4):
    input_dm_set.append(seperate(p, math.pi/2, math.pi))
    desired_output_dm_set.append(measure_1)


test_input_dm_set = []
test_desired_output_dm_set = []
for p in np.arange(0.02, 1/3, 0.02):
    test_input_dm_set.append(seperate(p, math.pi/2, math.pi))
    test_desired_output_dm_set.append(measure_0)
for p in np.arange(0.34, 1, 0.02):
    test_input_dm_set.append(seperate(p, math.pi/2, math.pi))
    test_desired_output_dm_set.append(measure_1)

train_set_size = len(input_dm_set)
test_set_size = len(test_input_dm_set)


def loss(output_dm, desired_output_dm):
    return -((output_dm * desired_output_dm).tr())


def dl_df(output_dm, desired_output_dm, delta_output_dm):
    return -((delta_output_dm * desired_output_dm).tr()).real


def pH_pp(params):
    return [H_cm + H_cm.dag() for H_cm in H_COMPONENTS]


l = [[qt.qzero(N) for i in range(0, tot_param_num)] for j in range(0, tot_param_num)]


def pC_pp_list(params):
    for i in range(0, tot_param_num):
        l[i][i] = C_COMPONENTS[i]
    return l


def tot_solve(point_data, param):
    ctrl.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl)
    sr = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df, loss=loss)
    vlplc = vll.VariationalLearningPLC(mdl, ctrl, sr)
    input_dm = input_dm_set[point_data]
    desired_output_dm = desired_output_dm_set[point_data]
    output_dm = plc(input_dm)
    return vlplc._gradient(input_dm, desired_output_dm), 1-((output_dm * desired_output_dm).tr()), output_dm


def test_solve(point_data, param):
    ctrl.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl)
    input_dm = test_input_dm_set[point_data]
    desired_output_dm = test_desired_output_dm_set[point_data]
    output_dm = plc(input_dm)
    return (output_dm * desired_output_dm).tr(), (measure_0 * output_dm).tr(), (measure_1 * output_dm).tr()

ctrl = vll.Controller(times_list, parameters_list, time_invariant=time_inv_list, adjustable=adj_list)

pool = multiprocessing.Pool(processes=train_set_size)
result = []
for point in range(0, train_set_size):
    result.append(pool.apply_async(tot_solve, (point, parameters_list, )))
pool.close()
pool.join()

grad_ave = np.zeros((len(times_list), tot_param_num))
cost = 0
out_dm = []
for i in result:
    cost += i.get()[1]/train_set_size
    grad_ave += i.get()[0]/train_set_size

pool = multiprocessing.Pool(processes=test_set_size)
test_result = []
for point in range(0, test_set_size):
    test_result.append(pool.apply_async(test_solve, (point, parameters_list, )))
pool.close()
pool.join()

accuracy = 0
for i in test_result:
    accuracy += i.get()[0]/test_set_size

update_num = 0
cost_list = [cost]
accuracy_list = [accuracy]
update_list = [0]
update_tot = 150
rgb_list = []
while update_num < update_tot:
    learning_rate_h = 20
    learning_rate_gama = 20
    for t in range(0, len(times_list)):
        for i in range(0, h_num):
            parameters_list[t][i] = parameters_list[t][i] - learning_rate_h * grad_ave[t][i]
        for j in range(h_num, tot_param_num):
            parameters_list[t][j] = parameters_list[t][j] - learning_rate_gama * grad_ave[t][j]

    result = []
    pool = multiprocessing.Pool(processes=train_set_size)
    for point in range(0, train_set_size):
        result.append(pool.apply_async(tot_solve, (point, parameters_list, )))
    pool.close()
    pool.join()

    cost = 0
    grad_ave = np.zeros((len(times_list), tot_param_num))
    for i in result:
        cost += i.get()[1]/train_set_size
        grad_ave += i.get()[0]/train_set_size
    cost_list.append(cost)

    pool = multiprocessing.Pool(processes=test_set_size)
    test_result = []
    for point in range(0, test_set_size):
        test_result.append(pool.apply_async(test_solve, (point, parameters_list, )))
    pool.close()
    pool.join()

    accuracy = 0
    for i in test_result:
        accuracy += i.get()[0]/test_set_size
    accuracy_list.append(accuracy)

    if update_num == update_tot-1:
        for i in test_result:
            rgb_list.append([i.get()[1], 0, i.get()[2]])

    update_num += 1
    update_list.append(update_num)

print("rgb_list={}".format(rgb_list))
print("update_list={}".format(update_list))
print("cost_list={}".format(cost_list))
print("accuracy_list={}".format(accuracy_list))


