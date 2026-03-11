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
import copy
from scipy.interpolate import interp1d
from scipy.integrate import quad
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

test_sent = ['so dawn goes down to day', 'nothing gold can stay', 'i love to stay here until the dawn','i love to go out for love']
test_lower = [0 for i in range(0, len(test_sent))]
test_without_punctuation = [0 for j in range(0, len(test_sent))]
test_tokens = [0 for k in range(0, len(test_sent))]
test_tagged_sent = [0 for t in range(0, len(test_sent))]
test_lemmas_sent = [0 for l in range(0, len(test_sent))]
test_wordnet_pos = [0 for w in range(0, len(test_sent))]
test_without_stopwords = [0 for w in range(0, len(test_sent))]
test_cleaned_text = [0 for c in range(0, len(test_sent))]
test_freq = [0 for f in range(0, len(test_sent))]
test_freq_list = [0 for c in range(0, len(test_sent))]
for i in range(0, len(test_sent)):
    test_lower[i] = test_sent[i].lower()  # 转换为小写
    test_remove = str.maketrans('', '', string.punctuation)  # 去除标点
    test_without_punctuation[i] = test_lower[i].translate(test_remove)


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
    test_tokens[i] = nltk.word_tokenize(test_without_punctuation[i])   # 分词list
    test_tagged_sent[i] = pos_tag(test_tokens[i])
    wnl = WordNetLemmatizer()
    test_lemmas_sent[i] = []
    for tag in test_tagged_sent[i]:
        test_wordnet_pos[i] = get_wordnet_pos(tag[1]) or wordnet.NOUN
        test_lemmas_sent[i].append(wnl.lemmatize(tag[0], pos=test_wordnet_pos[i]))  # 词形还原
    test_without_stopwords[i] = [w for w in test_lemmas_sent[i] if not w in stopwords.words('english')]
    s = nltk.stem.SnowballStemmer('english')
    test_cleaned_text[i] = [s.stem(ws) for ws in test_without_stopwords[i]]
test_sents = len(test_sent)
test_words_in_sent = [0 for i in range(0, test_sents)]
for s in range(0, test_sents):
    test_words_in_sent[s] = len(test_cleaned_text[s])
test_input_num = len(test_sent)

t_input = 20  # 总输入时间
initial_learning_rate_h = 0.1
initial_learning_rate_g = 1
gama = 1


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

test_delta_t_for_s = []
for s in range(0, len(test_sent)):
    test_delta_t_for_s.append(int(t_input/test_words_in_sent[s]))

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


def test_input_liouv_for_words_in(ts):  # 得到每一句话输入对应的输出
    test_delta_t = test_delta_t_for_s[ts]
    test_delta_times_list = np.linspace(0, (test_delta_t-1)/2, test_delta_t)
    test_parameters_list = np.zeros((len(test_delta_times_list), int((N-3)*(N-4)/2)+3*(N-3)))
    test_parameters = [0 for i in range(0, len(cleaned_text_all))]
    test_ctrl_input = []
    test_super_operator = []  # 每个单词的超算子
    test_time_depend_S = qt.to_super(qt.identity(N))
    for tw in range(0, test_words_in_sent[ts]):
        for tt in range(0, len(test_parameters_list)):
            test_parameters_list[tt, labeled_freq_dict_all[test_cleaned_text[ts][tw]]+int((N-3)*(N-4)/2)-1] = gama
        test_parameters[tw] = copy.deepcopy(test_parameters_list)
        test_ctrl_input.append(vll.Controller(test_delta_times_list, test_parameters[tw], time_invariant=time_inv_list))
        test_super_operator.append(vll.ParameterizedLindbladChannel(mdl, test_ctrl_input[-1]).get_super_operator())   # 得到超算子
    test_time_order = np.linspace(test_words_in_sent[ts]-1, 0, test_words_in_sent[ts])  # 时序
    for tw in test_time_order:
        test_time_depend_S = test_time_depend_S * test_super_operator[int(tw)]  # 得到输入单词超算子时序乘积
    return test_time_depend_S


# 得到每句话输入后的密度矩阵列表output_dm_list_input
input_dm_input = qt.basis(N, 0)*qt.basis(N, 0).dag()
S_list = []
output_dm_list_input = []
for s in range(0, sents):
    S_list.append(input_liouv_for_words_in(s))
    output_dm_list_input.append(qt.vector_to_operator(S_list[-1] * qt.operator_to_vector(input_dm_input)))


measure = []
for neuron in range(1, N-3+1):
    measure.append(qt.basis(N, neuron) * qt.basis(N, neuron).dag())

input_sent_list = []
for s in range(0, len(language_sent)):
    input_sent_list.append([(measure[i]*output_dm_list_input[s]).tr() for i in range(0, N-3)])

test_output_dm_list_input = []
tS_list = []
for ts in range(0, len(test_sent)):
    tS_list.append(test_input_liouv_for_words_in(ts))
    test_output_dm_list_input.append(qt.vector_to_operator(tS_list[-1] * qt.operator_to_vector(input_dm_input)))

test_input_sent_list = []
for s in range(0, len(test_sent)):
    test_input_sent_list.append([(measure[i]*test_output_dm_list_input[s]).tr() for i in range(0, N-3)])


no_list = [6, 7, 8, 9, 10, 11]
yes_list = [0, 1, 2, 3, 4, 5]  # 标签
sample_num = 15
w_list_for_sample = []
b_no_for_sample = []
b_yes_for_sample = []
for sample in range(0, sample_num):
    w_list_1 = []
    for i in range(0, 2*(N-3)):
        w_list_1.append(random.uniform(-1, 1))
    bno = random.uniform(-1, 1)
    byes = random.uniform(-1, 1)
    w_list_for_sample.append(w_list_1)
    b_no_for_sample.append(bno)
    b_yes_for_sample.append(byes)


def evolution(parameter_w, parameter_bno, parameter_byes):
    w_no, w_yes = [], []
    for ii in range(0, 2*(N-3), 2):
        w_no.append(parameter_w[ii])
        w_yes.append(parameter_w[ii+1])

    z_no = [0 for s in range(0, len(language_sent))]
    z_yes = [0 for s in range(0, len(language_sent))]
    softmax_no = [0 for s in range(0, len(language_sent))]
    softmax_yes = [0 for s in range(0, len(language_sent))]

    for s in range(0, len(language_sent)):
        z_no[s] = sum(w * p for w, p in zip(w_no, input_sent_list[s])) + parameter_bno
        z_yes[s] = sum(w * p for w, p in zip(w_yes, input_sent_list[s])) + parameter_byes
        softmax_no[s] = 1/(1+math.exp(z_yes[s]-z_no[s]))
        softmax_yes[s] = 1/(1+math.exp(z_no[s]-z_yes[s]))

    no_sents = 0
    yes_sents = 0
    for ss in no_list:
        no_sents += softmax_no[ss]
    for ss in yes_list:
        yes_sents += softmax_yes[ss]
    cost = 1-(no_sents+yes_sents)/len(language_sent)

    pl_pw_list = []
    for w in range(0, N-3):  # 对于yes/no的某个参数
        pl_pw_no = 0
        pl_pbno = 0
        pl_pw_yes = 0
        pl_pbyes = 0
        for ss in range(0, len(language_sent)):  # 对所有连向yes/no的句子求和
            if ss in no_list:
                pl_pw_no += softmax_no[ss]*(softmax_no[ss]-1)*input_sent_list[ss][w]
                pl_pbno += softmax_no[ss]*(softmax_no[ss]-1)/len(language_sent)
            if ss in yes_list:
                pl_pw_yes += softmax_yes[ss]*(softmax_yes[ss]-1)*input_sent_list[ss][w]
                pl_pbyes += softmax_yes[ss]*(softmax_yes[ss]-1)/len(language_sent)

        pl_pw_list.append(pl_pw_no/(len(language_sent)))
        pl_pw_list.append(pl_pw_yes/(len(language_sent)))

    return cost, pl_pw_list, pl_pbno, pl_pbyes


def test_evolution(parameter_w, parameter_bno, parameter_byes):
    w_no, w_yes = [], []
    for ii in range(0, 2*(N-3), 2):
        w_no.append(parameter_w[ii])
        w_yes.append(parameter_w[ii+1])
    z_no = [0 for s in range(0, len(test_sent))]
    z_yes = [0 for s in range(0, len(test_sent))]
    softmax_yes = [0 for s in range(0, len(test_sent))]
    for s in range(0, len(test_sent)):
        z_no[s] = sum(w * p for w, p in zip(w_no, test_input_sent_list[s])) + parameter_bno
        z_yes[s] = sum(w * p for w, p in zip(w_yes, test_input_sent_list[s])) + parameter_byes
        softmax_yes[s] = 1/(1+math.exp(z_no[s]-z_yes[s]))
    return softmax_yes  # 准确率


def update_for_sample(sam):
    w_list = w_list_for_sample[sam]
    b_no = b_no_for_sample[sam]
    b_yes = b_yes_for_sample[sam]
    result_for_sample = evolution(w_list, b_no, b_yes)
    loss_list = [result_for_sample[0]]
    pl_pw = result_for_sample[1]
    pl_pb_no = result_for_sample[2]
    pl_pb_yes = result_for_sample[3]

    test_result_for_sample = test_evolution(w_list, b_no, b_yes)
    accuracy_for_sent1_for_sample = [test_result_for_sample[0]]
    accuracy_for_sent2_for_sample = [test_result_for_sample[1]]
    accuracy_for_sent3_for_sample = [test_result_for_sample[2]]
    accuracy_for_sent4_for_sample = [test_result_for_sample[3]]


    update_num = 0
    while update_num < 200:
        rate_w_0 = 3
        rate_b_0 = 3
        UPDATE_NUM = 15
        if update_num < 100 or update_num == 100:
            rate_w = rate_w_0/(1+update_num/UPDATE_NUM)
            rate_b = rate_b_0/(1+update_num/UPDATE_NUM)
        else:
            rate_w = rate_w_0/(1+100/UPDATE_NUM)
            rate_b = rate_b_0/(1+100/UPDATE_NUM)
        for ii in range(0, 2*(N-3)):
            w_list[ii] = w_list[ii] - rate_w * pl_pw[ii]
        b_no = b_no - rate_b * pl_pb_no
        b_yes = b_yes - rate_b * pl_pb_yes
        result_for_sample = evolution(w_list, b_no, b_yes)
        loss = result_for_sample[0]
        pl_pw = result_for_sample[1]
        pl_pb_no = result_for_sample[2]
        pl_pb_yes = result_for_sample[3]
        test_result_for_sample = test_evolution(w_list, b_no, b_yes)
        accuracy_for_sent1_for_sample.append(test_result_for_sample[0])
        accuracy_for_sent2_for_sample.append(test_result_for_sample[1])
        accuracy_for_sent3_for_sample.append(test_result_for_sample[2])
        accuracy_for_sent4_for_sample.append(test_result_for_sample[3])
        update_num = update_num + 1
        loss_list.append(loss)
    return loss_list, accuracy_for_sent1_for_sample, accuracy_for_sent2_for_sample, accuracy_for_sent3_for_sample, accuracy_for_sent4_for_sample

# update_list = np.linspace(0, 20, 21)
# plt.plot(update_list, loss_list)
# plt.show()


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=1)
    result = []
    for sample in range(0, sample_num):
        result.append(pool.apply_async(update_for_sample, (sample,)))
    pool.close()
    pool.join()
    loss_list_for_sample_list = []
    accuracy_for_sent1_for_sample_list = []
    accuracy_for_sent2_for_sample_list = []
    accuracy_for_sent3_for_sample_list = []
    accuracy_for_sent4_for_sample_list = []
    for i in result:
        loss_list_for_sample_list.append(i.get()[0])
        accuracy_for_sent1_for_sample_list.append(i.get()[1])
        accuracy_for_sent2_for_sample_list.append(i.get()[2])
        accuracy_for_sent3_for_sample_list.append(i.get()[3])
        accuracy_for_sent4_for_sample_list.append(i.get()[4])
    # Loss = open('D:\Program Files (x86)\PyCharm 5.0.3\project\classicalNN-poem\classicalNN_loss.txt', mode='w')
    # Loss.write(str(loss_list_for_sample_list))
    # sent1 = open('D:\Program Files (x86)\PyCharm 5.0.3\project\classicalNN-poem\classicalNN_test1.txt', mode='w')
    # sent1.write(str(accuracy_for_sent1_for_sample_list))
    # sent2 = open('D:\Program Files (x86)\PyCharm 5.0.3\project\classicalNN-poem\classicalNN_test2.txt', mode='w')
    # sent2.write(str(accuracy_for_sent2_for_sample_list))
    # sent3 = open('D:\Program Files (x86)\PyCharm 5.0.3\project\classicalNN-poem\classicalNN_test3.txt', mode='w')
    # sent3.write(str(accuracy_for_sent3_for_sample_list))
    # sent4 = open('D:\Program Files (x86)\PyCharm 5.0.3\project\classicalNN-poem\classicalNN_test4.txt', mode='w')
    # sent4.write(str(accuracy_for_sent4_for_sample_list))
