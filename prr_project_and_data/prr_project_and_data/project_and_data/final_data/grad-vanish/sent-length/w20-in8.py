import multiprocessing
import numpy as np
import qutip as qt
import qutils as qu
import time
import valedian0910 as vll
from qutip import *
import random
import math
import cmath
import time
import copy
import matplotlib.pyplot as plt
import string
from scipy.interpolate import interp1d
from scipy.integrate import quad

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

language_sent = ['a b c d e f g h']

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
#     without_stopwords[i] = [w for w in lemmas_sent[i] if not w in stopwords.words('english')]
#     s = nltk.stem.SnowballStemmer('english')
#     cleaned_text[i] = [s.stem(ws) for ws in without_stopwords[i]]  # 去除停止词后的list
    cleaned_text[i] = [ws for ws in lemmas_sent[i]]
# print(cleaned_text)
labeled_freq_list_all = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7), ('h', 8), ('i', 9), ('j', 10),
                         ('k', 11), ('l', 12), ('m', 13), ('n', 14), ('o', 15), ('p', 16), ('r', 17), ('s', 18), ('t', 19), ('u', 20)]

labeled_freq_dict_all = dict(labeled_freq_list_all)

sents = len(language_sent)  # 句子个数
input_num = len(language_sent)
N = len(labeled_freq_list_all)+3

random_gama_lower_bound = -1
random_gama_upper_bound = 1

words_in_sent = [0 for i in range(0, sents)]
for s in range(0, sents):
    words_in_sent[s] = len(cleaned_text[s])  # 每个句子有多少单词[4, 3, 2, 4]

t_input = 200  # 总输入时间
t_unitary = 20
t_output = 20


# 哈密顿量连接系数
h = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(0, int((N-3)*(N-4)/2)):
    # h[i] = random.uniform(0, 10)
    h[i] = 0
# print(h)
# 输入连接系数
gama = 0.15
# 输出随机初始化
# gama_out = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
# for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
#     # gama_out[i] = random.uniform(0, 0.5)
#     gama_out[i] = 0.1
# gama_out = [0, 0, 0, 0, 0.24488312652484226, 0.03661176503267891, 0.0374070343704571, 0.23722418132040524]

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
    parameters = [0 for i in range(0, words_in_sent[s])]
    ctrl_input = []
    super_operator = []  # 每个单词的超算子
    time_depend_S = qt.to_super(qt.identity(N))
    parameters_list = np.zeros((len(delta_times_list), int((N-3)*(N-4)/2)+3*(N-3)))
    for w in range(0, words_in_sent[s]):
        for t in range(0, len(parameters_list)):
            parameters_list[t, labeled_freq_dict_all[cleaned_text[s][w]]+int((N-3)*(N-4)/2)-1] = gama
        # for i in range(0, int((N-3)*(N-4)/2)):
        #     for t in range(0, len(parameters_list)):
        #         parameters_list[t, i] = h[i]           # 控制哈密顿算子连接
        parameters[w] = copy.deepcopy(parameters_list)
        ctrl_input.append(vll.Controller(delta_times_list, parameters[w], time_invariant=time_inv_list))
        super_operator.append(vll.ParameterizedLindbladChannel(mdl, ctrl_input[-1]).get_super_operator())
    time_order = np.linspace(words_in_sent[s]-1, 0, words_in_sent[s])
    for w in time_order:
        time_depend_S = time_depend_S * super_operator[int(w)]
    return time_depend_S

# print(input_liouv_for_words_in(0))


def input_evolution(p, s):  # 解演化法求输入阶段输出
    ctrl = ctrl_input_list[s]
    ctrl.safe_set('parameters', p)
    plc_input = vll.ParameterizedLindbladChannel(
        mdl,
        ctrl
    )
    input_output_dm = plc_input(input_dm_input)
    return input_output_dm


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


def output_evolution(parameters_list, input_dm, desired_output_dm, get_grad):
    times_list_output = np.linspace(0, (t_output-1)/2, t_output)
    ctrl_output = vll.Controller(times_list_output, parameters_list, time_invariant=time_inv_list, adjustable=adj_output)
    if get_grad:
        sr2 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_2, loss=loss_2)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_output, sr2)
        return vlplc._gradient(input_dm, desired_output_dm)
    else:
        # return -((output_dm * desired_output_dm).tr())
        return output_dm


def tot_solve(s, parameters_list_o):
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_input[s], desired_output_list[s], get_grad=True)
    grad_ave_o = result_output_evolution_list
    return grad_ave_o


adj_unitary = []   # 更新时只有哈密顿连接能够被改变
for i in range(0, int((N-3)*(N-4)/2)):
    adj_unitary.append(i)
adj_output = []
for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):  # 更新时只有输出lindblad算子能够被改变
    adj_output.append(i)


def get_grad_ave_for_sample():
    gama_out = [0 for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))]  # 每个sample重置一次随机gama
    for ii in range(int((N-3)*(N-4)/2)+(N-3), int((N-3)*(N-4)/2)+3*(N-3)):
        gama_out[ii] = random.uniform(random_gama_lower_bound, random_gama_upper_bound)

    times_list_output = np.linspace(0, (t_output-1)/2, t_output)
    parameters_list_output = np.zeros((t_output, int((N-3)*(N-4)/2)+3*(N-3)))

    for t in range(0, t_output):
        for ii in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
            parameters_list_output[t, ii] = gama_out[ii]
    result_in = []
    for sent in range(0, sents):
        result_in.append(tot_solve(sent, parameters_list_output))

    grad_ave_output = np.zeros((t_output, int((N-3)*(N-4)/2)+3*(N-3)))
    for ii in result_in:
        grad_ave_output += ii/input_num
    grad_time_average_o = quad(interp1d(
                times_list_output,
                grad_ave_output[:, int((N-3)*(N-4)/2)+(N-3)],
                kind='cubic'
            ), times_list_output[0], times_list_output[-1])[0]/times_list_output[-1]    # 积分平均
    return grad_time_average_o


# 得到每句话输入后的密度矩阵列表output_dm_list_input
input_dm_input = qt.basis(N, 0)*qt.basis(N, 0).dag()
S_list = []
output_dm_list_input = []
for s in range(0, sents):
    S_list.append(input_liouv_for_words_in(s))
    output_dm_list_input.append(qt.vector_to_operator(S_list[-1] * qt.operator_to_vector(input_dm_input)))


measure1 = basis(N, N-1)*basis(N, N-1).dag()
measure0 = basis(N, N-2)*basis(N, N-2).dag()
desired_output_list = [measure1]

process_num = 24
sample_num = 1000


if __name__ == "__main__":
    pool_grad = multiprocessing.Pool(processes=process_num)
    result = []
    grad_abs_list = []
    grad_list = []
    for sample in range(0, sample_num):
        result.append(pool_grad.apply_async(get_grad_ave_for_sample, ()))
    pool_grad.close()
    pool_grad.join()
    for i in result:
        grad_list.append(i.get())
    grad_ave_for_sample = open('/share/home/sjwu/wlj/project/3-layers/grad-vanish/data/w20-in8.txt', mode='w')
    grad_ave_for_sample.write(str(grad_list))








