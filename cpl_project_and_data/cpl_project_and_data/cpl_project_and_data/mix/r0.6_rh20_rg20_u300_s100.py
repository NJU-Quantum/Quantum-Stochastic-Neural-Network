import numpy as np
import qutip as qt
from qutip import *
import math
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import copy
import valedian0910 as vll
import multiprocessing
from scipy.linalg import sqrtm

# 模型
in_layer = 2
hidden_layer = 2
out_layer = 2
N = sum([in_layer, hidden_layer, out_layer])
H_COMPONENTS = []
for in_neuron_1 in range(0, in_layer):  # 输入层之中
    for in_neuron_2 in range(0, in_layer):
        if in_neuron_1 < in_neuron_2:
            H_COMPONENTS.append(qt.basis(N, in_neuron_1) * qt.basis(N, in_neuron_2).dag())


for in_neuron in range(0, in_layer):   # 输入层到隐藏层
    for hid1_neuron in range(in_layer, in_layer+hidden_layer):
        H_COMPONENTS.append(qt.basis(N, hid1_neuron) * qt.basis(N, in_neuron).dag())


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


def mixed(theta, phi, r):
    rho = (qt.identity(2) + r * (math.sin(theta)*math.cos(phi)*qt.sigmax() + math.sin(theta)*math.sin(phi)*qt.sigmay() + math.cos(theta)*qt.sigmaz()))/2
    return rho

state_pair_num = 100
sample_list = []
theta_list = []
angle_list = []
for i in range(state_pair_num):
    theta1 = random.uniform(0, math.pi)
    phi1 = random.uniform(0, 2*math.pi)
    theta2 = random.uniform(0, math.pi)
    phi2 = random.uniform(0, 2*math.pi)
    sample = [mixed(theta1, phi1, 0.6), mixed(theta2, phi2, 0.6)]
    angle_list.append([theta1, phi1, theta2, phi2])
    sample_list.append(sample)
    qsave(sample_list, '/share/home/sjwu/wlj/project/state/data/random-mix-sample-list-r0.6-v1')   # 保存所有样本


train_set_for_sample = []
input_dm_for_sample = []
p_error_sample_list = []
train_set_size = 2
for state_pair in sample_list:
    train_set = [state_pair[0][0, 0]*basis(2, 0)*basis(2, 0).dag() + state_pair[0][0, 1]*basis(2, 0)*basis(2, 1).dag() + state_pair[0][1, 0]*basis(2, 1)*basis(2, 0).dag() + state_pair[0][1, 1]*basis(2, 1)*basis(2, 1).dag(),
                 state_pair[1][0, 0]*basis(2, 0)*basis(2, 0).dag() + state_pair[1][0, 1]*basis(2, 0)*basis(2, 1).dag() + state_pair[1][1, 0]*basis(2, 1)*basis(2, 0).dag() + state_pair[1][1, 1]*basis(2, 1)*basis(2, 1).dag()]

    p_error = (1-(state_pair[0]-state_pair[1]).norm()/2)/2
    p_error_sample_list.append(p_error)
    input_dm_for_sample.append([state_pair[0][0, 0]*basis(N, 0)*basis(N, 0).dag() + state_pair[0][0, 1]*basis(N, 0)*basis(N, 1).dag() + state_pair[0][1, 0]*basis(N, 1)*basis(N, 0).dag() + state_pair[0][1, 1]*basis(N, 1)*basis(N, 1).dag(),
                                state_pair[1][0, 0]*basis(N, 0)*basis(N, 0).dag() + state_pair[1][0, 1]*basis(N, 0)*basis(N, 1).dag() + state_pair[1][1, 0]*basis(N, 1)*basis(N, 0).dag() + state_pair[1][1, 1]*basis(N, 1)*basis(N, 1).dag()])

# print(input_dm_for_sample)
print("angle_list={}".format(angle_list))
print('p_error_sample_list=%s' % p_error_sample_list)
print('mean_p_error=%s' % np.mean(p_error_sample_list))


measure_0 = basis(N, N-2)*basis(N, N-2).dag()
measure_1 = basis(N, N-1)*basis(N, N-1).dag()
dic = {0: measure_0, 1: measure_1}

desired_output_dm_set = [measure_0,
                         measure_1]


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


def tot_solve(point_data, param, s):
    ctrl = vll.Controller(times_list, param, time_invariant=time_inv_list)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl)
    sr = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df, loss=loss)
    vlplc = vll.VariationalLearningPLC(mdl, ctrl, sr)
    input_dm_list = input_dm_for_sample[s]
    input_dm = input_dm_list[point_data]
    desired_output_dm = desired_output_dm_set[point_data]
    output_dm = plc(input_dm)
    return vlplc._gradient(input_dm, desired_output_dm), 1-((output_dm * desired_output_dm).tr())

times_list = np.linspace(0, t_tot, 2*t_tot+1)
learning_rate_h = 20
learning_rate_gama = 20
update_tot = 300

# 初始化参数
h_initial = [random.uniform(0, 2) for i in range(h_num)]
# h_initial = [0.0942534525960201, 0.023785361513672033, 0.06874517533498006, 0.1508755150420773, 0.1837458286682511]
print('h_initial=%s' % h_initial)
gama_initial = [0.5 for i in range(tot_param_num)]


def evolution_for_sample(s):
    parameters_list = np.zeros((len(times_list), tot_param_num))
    for i in range(0, h_num):
        parameters_list[:, i] = h_initial[i]

    for i in range(h_num, tot_param_num):
        parameters_list[:, i] = gama_initial[i]

    result = []
    for point in range(0, train_set_size):
        result.append(tot_solve(point, parameters_list, s))
    grad_ave = np.zeros((len(times_list), tot_param_num))
    cost = 0
    for i in result:
        cost += i[1]/train_set_size
        grad_ave += i[0]/train_set_size

    update_num = 0
    cost_list = [cost]

    while update_num < update_tot:
        for t in range(0, len(times_list)):
            for i in range(0, h_num):
                parameters_list[t][i] = parameters_list[t][i] - learning_rate_h * grad_ave[t][i]
            for j in range(h_num, tot_param_num):
                parameters_list[t][j] = parameters_list[t][j] - learning_rate_gama * grad_ave[t][j]

        result = []
        for point in range(0, train_set_size):
            result.append(tot_solve(point, parameters_list, s))

        cost = 0
        grad_ave = np.zeros((len(times_list), tot_param_num))
        for i in result:
            cost += i[1]/train_set_size
            grad_ave += i[0]/train_set_size
        cost_list.append(cost)
        update_num += 1
    return cost_list


if __name__ == "__main__":
    result_sample = []
    pool = multiprocessing.Pool(processes=96)
    for sam in range(0, state_pair_num):
        result_sample.append(pool.apply_async(evolution_for_sample, (sam, )))
    pool.close()
    pool.join()

    cost_list_sample = []
    for i in result_sample:
        cost_list_sample.append(i.get())
    print('cost_list_sample=%s' % cost_list_sample)

    cost_list_sample_for_update = [[] for i in range(0, update_tot)]
    cost_list_sample_ave = [0 for i in range(0, update_tot)]
    cost_list_sample_var = [0 for i in range(0, update_tot)]
    for update_num in range(0, update_tot):
        for sample in cost_list_sample:
            cost_list_sample_for_update[update_num].append(sample[update_num])
        cost_list_sample_ave[update_num] = np.mean(cost_list_sample_for_update[update_num])
        cost_list_sample_var[update_num] = np.var(cost_list_sample_for_update[update_num])
    print('cost_list_sample_ave=%s' % cost_list_sample_ave)
    print('cost_list_sample_var=%s' % cost_list_sample_var)


