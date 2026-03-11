import multiprocessing
import numpy as np
import qutip as qt
import valedian211113 as vll
from qutip import *
import random
import copy
import string
from scipy.interpolate import interp1d
from scipy.integrate import quad

import nltk
import nltk.stem
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


np.set_printoptions(threshold=np.inf)

"""
We investigate the effect of the word sequence length the gradient.
We calculate the gradient of the first parameters in the cases of the x_train containing a sequence of words of different length.
It is an example of that the sequence in the x_train contains 2 words.
We substitute a letter for a word.
"""

"""
Set the training set
"""

x_train = ['a b']


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


def pre_process(word_sequence_list):
    """
    The pro-process for the input language data.
    It ill lemmatize the input words to their normal form and delete stop words.
    :param word_sequence_list: x_train or x_test
    :return word sequences with out stop words [['gold', 'sun', 'dawn'], ['love', 'stay', 'day', 'sun'], ...]
    """
    cleaned_data = []
    for sequence in word_sequence_list:
        # lower
        lower = sequence.lower()
        # remove punctuation
        remove = str.maketrans('', '', string.punctuation)
        without_punctuation = lower.translate(remove)
        # restore the word to its original form
        tokens = nltk.word_tokenize(without_punctuation)
        tagged_sent = pos_tag(tokens)
        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        cleaned_sequence = [ws for ws in lemmas_sent]
        cleaned_data.append(cleaned_sequence)
    return cleaned_data

# information of the data
cleaned_text = pre_process(x_train)
train_size = len(x_train)
words_in_se = [len(cleaned_text[s]) for s in range(0, train_size)]  # number of words in each sequence

"""
Encode the word into the neurons
"""

encode_word_list = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7), ('h', 8), ('i', 9), ('j', 10),
                    ('k', 11), ('l', 12), ('m', 13), ('n', 14), ('o', 15), ('p', 16), ('r', 17), ('s', 18), ('t', 19), ('u', 20)]

encode_word_dict = dict(encode_word_list)
N = len(encode_word_list)+3  # the total numbers of the neurons

# evolution time
t_input = 200  
t_unitary = 20
t_output = 20
delta_t_for_s = [(int(t_input/words_in_se[s])) for s in range(0, train_size)]


# Hamiltonian
h = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(0, int((N-3)*(N-4)/2)):
    h[i] = 0
# input Linblad operator
gamma_in = 0.15
# all the parameters are time independent
time_indep_params_list = list(range(int((N-3)*(N-4)/2)+3*(N-3)))

"""
Construct quantum neural networks
"""

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
evolution (1). input
The input evolution will not be optimized.
"""


def input_liouv_for_words_in(text, s):
    """

    :param s: index of the sequence
    :return: super operators for the sequence
    """
    word_num = len(text[s])
    delta_t = int(t_input/word_num)
    delta_times_list = np.linspace(0, (delta_t-1)/2, delta_t)
    parameters_list = np.zeros((len(delta_times_list), int((N-3)*(N-4)/2)+3*(N-3)))
    parameters = [0 for i in range(0, len(encode_word_list))]
    ctrl_input = []
    super_operator = []
    time_depend_S = qt.to_super(qt.identity(N))
    for w in range(0, len(text[s])):
        for t in range(0, len(parameters_list)):
            parameters_list[t, encode_word_dict[text[s][w]]+int((N-3)*(N-4)/2)-1] = gamma_in
        parameters[w] = copy.deepcopy(parameters_list)
        ctrl_input.append(vll.Controller(delta_times_list, parameters[w], time_invariant=time_indep_params_list))
        super_operator.append(vll.ParameterizedLindbladChannel(mdl, ctrl_input[-1]).get_super_operator())
    time_order = np.linspace(len(text[s])-1, 0, len(text[s]))
    for w in time_order:
        time_depend_S = time_depend_S * super_operator[int(w)]
    return time_depend_S


# the initial state of the QSNN
rho_0 = qt.basis(N, 0)*qt.basis(N, 0).dag()
# get rho_in
rho_in_list = [qt.vector_to_operator(input_liouv_for_words_in(cleaned_text, s) * qt.operator_to_vector(rho_0)) for s in range(train_size)]

"""
Define the loss and gradient
"""


def loss_output(output_dm, desired_output_dm):
    return -((output_dm * desired_output_dm).tr())


def dl_df_output(output_dm, desired_output_dm, delta_output_dm):
    return -((delta_output_dm * desired_output_dm).tr()).real


def pH_pp(params):
    return [H_cm + H_cm.dag() for H_cm in H_COMPONENTS]


def pC_pp_list(params):
    l = [[qt.qzero(N) for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))] for j in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
    for i in range(0, int((N-3)*(N-4)/2+3*(N-3))):
        l[i][i] = C_COMPONENTS[i]
    return l


def output_evolution(parameters_list, input_dm, desired_output_dm):
    """
    evolution (3). output
    The gradient of the parameters is evaluated simultaneously after each evolution.
    :return:
    """
    times_list_output = np.linspace(0, (t_output-1)/2, t_output)
    ctrl_output = vll.Controller(times_list_output, parameters_list, time_invariant=time_indep_params_list, adjustable=list(range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3))))
    sr2 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_output, loss=loss_output)
    vlplc = vll.VariationalLearningPLC(mdl, ctrl_output, sr2)
    return vlplc._gradient(input_dm, desired_output_dm)

 
def tot_solve(s, parameters_list_o):
    result_output_evolution_list = output_evolution(parameters_list_o, rho_in_list[s], desired_output_list[s])
    grad_ave_o = result_output_evolution_list
    return grad_ave_o


def get_grad_ave_for_sample():
    # parameters
    gamma_out = [0 for i in range(0, int((N-3)*(N-4)/2)+3*(N-3))]
    for ii in range(int((N-3)*(N-4)/2)+(N-3), int((N-3)*(N-4)/2)+3*(N-3)):
        gamma_out[ii] = random.uniform(-1, 1)

    times_list_output = np.linspace(0, (t_output-1)/2, t_output)
    parameters_list_output = np.zeros((t_output, int((N-3)*(N-4)/2)+3*(N-3)))
    for t in range(0, t_output):
        for ii in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
            parameters_list_output[t, ii] = gamma_out[ii]

    # evolution
    result_in = []
    for sent in range(0, train_size):
        result_in.append(tot_solve(sent, parameters_list_output))

    # calculate the time average of the gradient
    grad_ave_output = np.zeros((t_output, int((N-3)*(N-4)/2)+3*(N-3)))
    for ii in result_in:
        grad_ave_output += ii/train_size
    grad_time_average_o = quad(interp1d(
                times_list_output,
                grad_ave_output[:, int((N-3)*(N-4)/2)+(N-3)],
                kind='cubic'
            ), times_list_output[0], times_list_output[-1])[0]/times_list_output[-1]
    return grad_time_average_o

# labels
neuron_yes = basis(N, N-1)*basis(N, N-1).dag()
neuron_no = basis(N, N-2)*basis(N, N-2).dag()
desired_output_list = [neuron_yes]

process_num = 24
sample_num = 1000

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

grad_ave_for_sample = open('gradient_vanish_2words.txt', mode='w')
grad_ave_for_sample.write(str(grad_list))








