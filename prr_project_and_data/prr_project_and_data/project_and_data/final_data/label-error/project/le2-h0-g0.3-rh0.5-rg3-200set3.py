import multiprocessing
import numpy as np
import qutip as qt
import qutils as qu
import time
import valedian0910 as vll
from qutip import *
import random
from scipy.interpolate import interp1d
from scipy.integrate import quad
import math
import cmath
import copy
import matplotlib.pyplot as plt
import string

import nltk
from nltk.corpus import stopwords
import nltk.stem
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
np.set_printoptions(threshold=np.inf)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# start = time.clock()
# language_sent = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
#                  'sun gold day i', 'nothing go dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']
language_sent = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
                 'sun gold day i', 'day nothing dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']  # set3
lower = [0 for i in range(0, len(language_sent))]
without_punctuation = [0 for j in range(0, len(language_sent))]
tokens = [0 for k in range(0, len(language_sent))]
tagged_sent = [0 for t in range(0, len(language_sent))]
lemmas_sent = [0 for l in range(0, len(language_sent))]
wordnet_pos = [0 for w in range(0, len(language_sent))]
without_stopwords = [0 for w in range(0, len(language_sent))]
cleaned_text = [0 for c in range(0, len(language_sent))]
freq = [0 for f in range(0, len(language_sent))]
freq_list = [0 for c in range(0, len(language_sent))]
for i in range(0, len(language_sent)):
    lower[i] = language_sent[i].lower()  # 转换为小写
    remove = str.maketrans('', '', string.punctuation)  # 去除标点
    without_punctuation[i] = lower[i].translate(remove)

    tokens[i] = nltk.word_tokenize(without_punctuation[i])   # 分词list
# print(tokens)

    tagged_sent[i] = pos_tag(tokens[i])
    wnl = WordNetLemmatizer()
    lemmas_sent[i] = []
    for tag in tagged_sent[i]:
        wordnet_pos[i] = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent[i].append(wnl.lemmatize(tag[0], pos=wordnet_pos[i]))  # 词形还原
# print(lemmas_sent)
# 去除停止词
    without_stopwords[i] = [w for w in lemmas_sent[i] if not w in stopwords.words('english')]
    s = nltk.stem.SnowballStemmer('english')
    cleaned_text[i] = [s.stem(ws) for ws in without_stopwords[i]]  # 去除停止词后的list
labeled_freq_list_all = [('gold', 1), ('sun', 2), ('dawn', 3), ('love', 4), ('stay', 5), ('day', 6), ('go', 7), ('noth', 8)]
cleaned_text_all = ['gold', 'sun', 'dawn', 'love', 'stay', 'day', 'go', 'noth']
labeled_freq_dict_all = dict(labeled_freq_list_all)

sents = len(language_sent)  # 句子个数
input_num = len(language_sent)
N = len(labeled_freq_list_all)+3

words_in_sent = [0 for i in range(0, sents)]
for s in range(0, sents):
    words_in_sent[s] = len(cleaned_text[s])  # 每个句子有多少单词[4, 3, 2, 4]

t_input = 20  # 总输入时间
t_unitary = 20
t_output = 20


# 哈密顿量连接系数
h = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(0, int((N-3)*(N-4)/2)):
    # h[i] = random.uniform(0, 10)
    h[i] = 0
# print(h)
# 输入连接系数
gama = 1
# 输出随机初始化
gama_out = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
    # gama_out[i] = random.uniform(0, 10)
    gama_out[i] = 0.3


H_COMPONENTS = [qt.qzero(N) for h_components in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
C_COMPONENTS = [qt.qzero(N) for c_components in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
h_component = 0
while h_component < (N-3)*(N-4)/2:
    for line in range(1, N-3):
        for col in range(2, N-2):
            if col > line:
                H_COMPONENTS[h_component] = qt.basis(N, line) * qt.basis(N, col).dag()
                h_component = h_component+1
# print(H_COMPONENTS)
c_component = int((N-3)*(N-4)/2)
while c_component < (N-3)*(N-4)/2+3*(N-3):
    for line in range(1, N-2):
        C_COMPONENTS[c_component] = qt.basis(N, line) * qt.basis(N, 0).dag()
        c_component = c_component+1
    for col in range(1, N-2):
        C_COMPONENTS[c_component] = qt.basis(N, N-2) * qt.basis(N, col).dag()
        c_component = c_component+1
    for col in range(1, N-2):
        C_COMPONENTS[c_component] = qt.basis(N, N-1) * qt.basis(N, col).dag()
        c_component = c_component+1
# print(C_COMPONENTS)


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

delta_t_for_s = []
for s in range(0, sents):
    delta_t_for_s.append(int(t_input/words_in_sent[s]))


time_inv_list = []   # 所有参数时间不变
for i in range(0, int((N-3)*(N-4)/2)+3*(N-3)):
    time_inv_list.append(i)


def input_liouv_for_words_in(s):  # 得到每一句话输入对应的输出
    delta_t = delta_t_for_s[s]
    delta_times_list = np.linspace(0, (delta_t-1)/2, delta_t)
    parameters_list = np.zeros((len(delta_times_list), int((N-3)*(N-4)/2)+3*(N-3)))
    parameters = [0 for i in range(0, len(cleaned_text_all))]
    ctrl_input = []
    super_operator = []  # 每个单词的超算子
    time_depend_S = qt.to_super(qt.identity(N))
    for w in range(0, words_in_sent[s]):
        for t in range(0, len(parameters_list)):
            parameters_list[t, labeled_freq_dict_all[cleaned_text[s][w]]+int((N-3)*(N-4)/2)-1] = gama
        parameters[w] = copy.deepcopy(parameters_list)
        ctrl_input.append(vll.Controller(delta_times_list, parameters[w], time_invariant=time_inv_list))
        super_operator.append(vll.ParameterizedLindbladChannel(mdl, ctrl_input[-1]).get_super_operator())
    time_order = np.linspace(words_in_sent[s]-1, 0, words_in_sent[s])
    for w in time_order:
        time_depend_S = time_depend_S * super_operator[int(w)]
    return time_depend_S


def loss_2(output_dm, desired_output_dm):
    return -((output_dm * desired_output_dm).tr())


def loss_1(output_dm, desired_output_dm):
    return -((qt.vector_to_operator(super_operator_2 * qt.operator_to_vector(output_dm)) * desired_output_dm).tr())


def dl_df_2(output_dm, desired_output_dm, delta_output_dm):
    return -((delta_output_dm * desired_output_dm).tr()).real


def dl_df_1(output_dm, desired_output_dm, delta_output_dm):
    delta_output_dm_2 = qt.vector_to_operator(super_operator_2 * (qt.operator_to_vector(delta_output_dm)))
    return -((delta_output_dm_2 * desired_output_dm).tr()).real


def pH_pp(params):
    return [H_cm + H_cm.dag() for H_cm in H_COMPONENTS]


l = [[qt.qzero(N) for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))] for j in range(0, int((N-3)*(N-4)/2+3*(N-3)))]


def pC_pp_list(params):
    for i in range(0, int((N-3)*(N-4)/2+3*(N-3))):
        l[i][i] = C_COMPONENTS[i]
    return l


def unitary_evolution(parameters_list_u, input_dm, desired_output_dm, get_grad):
    ctrl_unitary.safe_set('parameters', parameters_list_u)
    plc_constant = vll.ParameterizedLindbladChannel(mdl, ctrl_unitary)
    output_dm = plc_constant(input_dm)
    if get_grad:
        sr1 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_1, loss=loss_1)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_unitary, sr1)
        return vlplc._gradient(input_dm, desired_output_dm), output_dm
    else:
        return output_dm


def output_evolution(parameters_list, input_dm, desired_output_dm, get_grad):
    ctrl_output.safe_set('parameters', parameters_list)
    plc_constant = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    output_dm = plc_constant(input_dm)
    if get_grad:
        sr2 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_2, loss=loss_2)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_output, sr2)
        return vlplc._gradient(input_dm, desired_output_dm), 1-((output_dm * desired_output_dm).tr())
    else:
        return 1-((output_dm * desired_output_dm).tr())


def get_super_operator_2(param):
    ctrl_output.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    super_2 = plc.get_super_operator()
    return super_2


def tot_solve(s, parameters_list_u, parameters_list_o):
    result_unitary_evolution_list = unitary_evolution(parameters_list_u, output_dm_list_input[s], desired_output_list[s], get_grad=True)
    output_dm_list_unitary = result_unitary_evolution_list[1]
    grad_ave_u = result_unitary_evolution_list[0]
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_unitary, desired_output_list[s], get_grad=True)
    c = result_output_evolution_list[1]
    grad_ave_o = result_output_evolution_list[0]
    return c, grad_ave_u, grad_ave_o


def label_error_tot_solve(s, parameters_list_u, parameters_list_o):
    result_unitary_evolution_list = unitary_evolution(parameters_list_u, output_dm_list_input[s], label_error_desired_output_list[s], get_grad=True)
    output_dm_list_unitary = result_unitary_evolution_list[1]
    grad_ave_u = result_unitary_evolution_list[0]
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_unitary, label_error_desired_output_list[s], get_grad=True)
    c = result_output_evolution_list[1]
    grad_ave_o = result_output_evolution_list[0]
    return c, grad_ave_u, grad_ave_o


# 酉演化参数
times_list_unitary = np.linspace(0, (t_unitary-1)/2, t_unitary)   # 输入时间 步长0.2
parameters_list_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
for i in range(0, int((N-3)*(N-4)/2)):
    for t in range(0, t_unitary):
        parameters_list_unitary[t, i] = h[i]  # 控制哈密顿算子连接
adj_unitary = []   # 更新时只有哈密顿连接能够被改变
for i in range(0, int((N-3)*(N-4)/2)):
    adj_unitary.append(i)
ctrl_unitary = vll.Controller(times_list_unitary, parameters_list_unitary, time_invariant=time_inv_list, adjustable=adj_unitary)

# 输出参数
times_list_output = np.linspace(0, (t_output-1)/2, t_output)   # 输入时间 步长0.2
parameters_list_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
for t in range(0, t_output):
    for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
        parameters_list_output[t, i] = gama_out[i]
adj_output = []
for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):  # 更新时只有输出lindblad算子能够被改变
    adj_output.append(i)
ctrl_output = vll.Controller(times_list_output, parameters_list_output, time_invariant=time_inv_list, adjustable=adj_output)


# 得到每句话输入后的密度矩阵列表output_dm_list_input
input_dm_input = qt.basis(N, 0)*qt.basis(N, 0).dag()
S_list = []
output_dm_list_input = []
for s in range(0, sents):
    S_list.append(input_liouv_for_words_in(s))
    output_dm_list_input.append(qt.vector_to_operator(S_list[-1] * qt.operator_to_vector(input_dm_input)))


measure1 = basis(N, N-1)*basis(N, N-1).dag()
measure0 = basis(N, N-2)*basis(N, N-2).dag()
desired_output_list = [measure0, measure0, measure1, measure1, measure1, measure1,
                       measure1, measure1, measure0, measure0, measure0, measure0]         # 有错误的标签
label_error_desired_output_list = [measure1, measure1, measure1, measure1, measure1, measure1,
                            measure0, measure0, measure0, measure0, measure0, measure0]     # 正确的标签


# 计算第一个梯度和代价
super_operator_2 = get_super_operator_2(parameters_list_output)
grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
cost = 0
grad_u_for_alltime = 0
grad_d_for_alltime = 0

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=input_num)
    result = []
    for s in range(0, sents):
        result.append(pool.apply_async(tot_solve, (s, parameters_list_unitary, parameters_list_output,)))
    pool.close()
    pool.join()
    for i in result:
        cost += i.get()[0]/input_num
        grad_ave_unitary += i.get()[1]/input_num
        grad_ave_output += i.get()[2]/input_num
        func_u = interp1d(times_list_unitary, sum(abs(grad_ave_unitary[:, i]) for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))), kind="cubic")
        grad_u_for_alltime = quad(func_u, times_list_unitary[0], times_list_unitary[-1])[0]
        func_d = interp1d(times_list_output, sum(abs(grad_ave_output[:, i]) for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))), kind="cubic")
        grad_d_for_alltime = quad(func_d, times_list_output[0], times_list_output[-1])[0]

cost_list = [cost]
update_list = [0]
update_num = 0
grad_u_for_alltime_list = [grad_u_for_alltime]
grad_d_for_alltime_list = [grad_d_for_alltime]

h_list = [[h[0]] for i in range(0, int((N-3)*(N-4)/2))]  # 记录所有哈密顿连接参数的更新
gama_list = [[gama_out[int((N-3)*(N-4)/2+N-3)]] for i in range(0, int(2*(N-3)))]
correct_step = 100

while update_num < 200:
    learning_rate_h_0 = 0.5
    learning_rate_gama_0 = 3
    UPDATE_NUM = 15
    if update_num < correct_step or update_num == correct_step:
        learning_rate_gama = learning_rate_gama_0/(1+update_num/UPDATE_NUM)
        learning_rate_h = learning_rate_h_0/(1+update_num/UPDATE_NUM)
    elif update_num > 2*correct_step:
        learning_rate_gama = learning_rate_gama_0/(1+2*correct_step/UPDATE_NUM)
        learning_rate_h = learning_rate_h_0/(1+2*correct_step/UPDATE_NUM)
    else:
        learning_rate_gama = learning_rate_gama_0/(1+(update_num-correct_step)/UPDATE_NUM)
        learning_rate_h = learning_rate_h_0/(1+(update_num-correct_step)/UPDATE_NUM)

    parameters_list_unitary = parameters_list_unitary - learning_rate_h * grad_ave_unitary
    parameters_list_output = parameters_list_output - learning_rate_gama * grad_ave_output

    super_operator_2 = get_super_operator_2(parameters_list_output)

    grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
    cost = 0
    grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
    if update_num < correct_step or update_num == correct_step:
        if __name__ == "__main__":
            pool = multiprocessing.Pool(processes=input_num)
            result = []
            for s in range(0, sents):
                result.append(pool.apply_async(tot_solve, (s, parameters_list_unitary, parameters_list_output,)))
            pool.close()
            pool.join()
            for i in result:
                cost += i.get()[0]/input_num
                grad_ave_unitary += i.get()[1]/input_num
                grad_ave_output += i.get()[2]/input_num
            func_u = interp1d(times_list_unitary, sum(abs(grad_ave_unitary[:, i]) for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))), kind="cubic")
            grad_u_for_alltime = quad(func_u, times_list_unitary[0], times_list_unitary[-1])[0]
            func_d = interp1d(times_list_output, sum(abs(grad_ave_output[:, i]) for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))), kind="cubic")
            grad_d_for_alltime = quad(func_d, times_list_output[0], times_list_output[-1])[0]
            update_num = update_num + 1
            cost_list.append(cost)
            grad_u_for_alltime_list.append(grad_u_for_alltime)
            grad_d_for_alltime_list.append(grad_d_for_alltime)
            update_list.append(update_num)

    else:
        if __name__ == "__main__":
            pool = multiprocessing.Pool(processes=input_num)
            result = []
            for s in range(0, sents):
                result.append(pool.apply_async(label_error_tot_solve, (s, parameters_list_unitary, parameters_list_output,)))
            pool.close()
            pool.join()
            for i in result:
                cost += i.get()[0]/input_num
                grad_ave_unitary += i.get()[1]/input_num
                grad_ave_output += i.get()[2]/input_num
            func_u = interp1d(times_list_unitary, sum(abs(grad_ave_unitary[:, i]) for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))), kind="cubic")
            grad_u_for_alltime = quad(func_u, times_list_unitary[0], times_list_unitary[-1])[0]
            func_d = interp1d(times_list_output, sum(abs(grad_ave_output[:, i]) for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))), kind="cubic")
            grad_d_for_alltime = quad(func_d, times_list_output[0], times_list_output[-1])[0]
            update_num = update_num + 1
            cost_list.append(cost)
            grad_u_for_alltime_list.append(grad_u_for_alltime)
            grad_d_for_alltime_list.append(grad_d_for_alltime)
            update_list.append(update_num)

file_handle = open('/share/home/sjwu/wlj/project/3-layers/label-error/data/le2-h0-g0.3.txt', mode='w')
file_handle.write(str(cost_list))
# print('update_list=%s' % update_list)
# print('grad_u_for_alltime_list=%s' % grad_u_for_alltime_list)
# print('grad_d_for_alltime_list=%s' % grad_d_for_alltime_list)
# print('parameters_list_unitary=%s' % parameters_list_unitary)
# print('parameters_list_output=%s' % parameters_list_output)





