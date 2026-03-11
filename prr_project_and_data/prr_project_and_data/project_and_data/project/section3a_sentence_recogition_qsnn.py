import numpy as np
import qutip as qt
import multiprocessing
import valedian211113 as vll
from qutip import *
import random
import copy
from scipy.interpolate import interp1d
from scipy.integrate import quad
import string
import nltk
from nltk.corpus import stopwords
import nltk.stem
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
np.set_printoptions(threshold=np.inf)

"""
It is the code used to train the QSNN to complete the sentence recognition task.
It is an example that there are only 2 words (2 neurons in the hidden layer) in our corpus.
We substitute a letter for a word.
We calculate its device robustness while training the model.
The training result is shown in section III. A and the result of the device robustness is shown in section IV. B.
It is an example to train the QSNNs with the specific initial value of the Hamiltonian (h=0.1) and with 1000 different sets of initial values of Lindblad operators.
"""

"""
Set the training set
"""

# Each training word sequence only contains 2 words.
x_train = ['b c', 'c b']

"""
Pre process the training and test data
"""


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
    :return word sequences with out stop words [['b', 'c'], ['c', 'b']]
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
        without_stopwords = [w for w in lemmas_sent if not w in stopwords.words('english')]
        s = nltk.stem.SnowballStemmer('english')
        cleaned_sequence = [s.stem(ws) for ws in without_stopwords]
        cleaned_data.append(cleaned_sequence)
    return cleaned_data

# information of the data
cleaned_text = pre_process(x_train)
train_size = len(x_train)
words_in_se = [len(cleaned_text[s]) for s in range(0, train_size)]  # number of words in each sequence

"""
Encode the word into the neurons
"""

encode_word_list = [('b', 1), ('c', 2)]
encode_word_dict = dict(encode_word_list)
N = len(encode_word_list)+3  # the total numbers of the neurons

"""
Initialization of the parameters
"""

# Hamiltonian
h = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(0, int((N-3)*(N-4)/2)):
    h[i] = 0.1
# input Linblad operator
gamma_in = 1
# output Lindblad operator
sample_num = 1000
gamma_out_list = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
gamma_out_for_for_sample = []
for sam in range(0, sample_num):
    for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
        gamma_out_list[i] = random.uniform(0, 1)
    gamma_out_for_for_sample.append(copy.deepcopy(gamma_out_list))

# all the parameters are time independent
time_indep_params_list = list(range(int((N-3)*(N-4)/2)+3*(N-3)))

# evolution time
t_input = 20
t_unitary = 20
t_output = 20
delta_t_for_s = [(int(t_input/words_in_se[s])) for s in range(0, train_size)]


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
# get rho_in for each sequence
rho_in_list = [qt.vector_to_operator(input_liouv_for_words_in(cleaned_text, s) * qt.operator_to_vector(rho_0)) for s in range(train_size)]

"""
Define the loss and gradient
"""


def loss_unitary(output_dm, desired_output_dm):
    global super_operator_unitary
    return -((qt.vector_to_operator(super_operator_unitary * qt.operator_to_vector(output_dm)) * desired_output_dm).tr())


def dl_df_unitary(output_dm, desired_output_dm, delta_output_dm):
    global super_operator_unitary
    delta_output_dm_2 = qt.vector_to_operator(super_operator_unitary * (qt.operator_to_vector(delta_output_dm)))
    return -((delta_output_dm_2 * desired_output_dm).tr()).real


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

"""
Define the evolution of the QSNN
"""


def unitary_evolution(parameters_list_u, input_dm, desired_output_dm, get_grad):
    """
    evolution (2). unitary evolution
    The gradient of the parameters is evaluated simultaneously after each evolution.
    :return:
    """
    global ctrl_unitary
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
    :return:
    """
    global ctrl_output
    ctrl_output.safe_set('parameters', parameters_list)
    plc_constant = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    output_dm = plc_constant(input_dm)
    if get_grad:
        sr2 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_output, loss=loss_output)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_output, sr2)
        return vlplc._gradient(input_dm, desired_output_dm), -((output_dm * desired_output_dm).tr())
    else:
        return -(output_dm * desired_output_dm).tr()


def get_super_operator_output(param):
    """
    :return: the super operator of the evolution (3) output
    """
    global ctrl_output
    ctrl_output.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    super_2 = plc.get_super_operator()
    return super_2


def tot_solve(s, parameters_list_u, parameters_list_o, get_robustness):
    """
    evolution of the training input and calculation of the gradient 
    :param s: input sequence index
    :return: loss, the gradient of \vec{h} and \vec{gamma}
    """
    times_list_output = np.linspace(0, (t_output-1)/2, t_output)
    result_unitary_evolution_list = unitary_evolution(parameters_list_u, rho_in_list[s], desired_rho_out_list[s], get_grad=True)
    output_dm_list_unitary = result_unitary_evolution_list[1]
    grad_ave_u = result_unitary_evolution_list[0]
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_unitary, desired_rho_out_list[s], get_grad=True)
    c = result_output_evolution_list[1]
    grad_ave_o = result_output_evolution_list[0]
    # evaluate the device robustness
    if get_robustness:
        func_grad_o = [0 for ii in range(int((N-3)*(N-4)/2)+N-3, int((N-3)*(N-4)/2)+3*(N-3))]
        robustness_o = [0 for ii in range(int((N-3)*(N-4)/2)+N-3, int((N-3)*(N-4)/2)+3*(N-3))]
        for ii in range(int((N-3)*(N-4)/2)+N-3, int((N-3)*(N-4)/2)+3*(N-3)):
            func_grad_o[ii-(int((N-3)*(N-4)/2)+N-3)] = interp1d(times_list_output, grad_ave_o[:, ii], kind="cubic")
            robustness_o[ii-(int((N-3)*(N-4)/2)+N-3)] = quad(func_grad_o[ii-(int((N-3)*(N-4)/2)+N-3)], times_list_output[0], times_list_output[-1])[0]/times_list_output[-1]  # 对每个参数的梯度按时间积分求平均
        robustness_for_all_g = sum([abs(rob)**2 for rob in robustness_o])/(2*(N-3))   # 对所有参数加和求平均
        return c, grad_ave_u, grad_ave_o, robustness_for_all_g
    else:
        return c, grad_ave_u, grad_ave_o

neuron_yes = basis(N, N-1)*basis(N, N-1).dag()
neuron_no = basis(N, N-2)*basis(N, N-2).dag()
desired_rho_out_list = [neuron_yes,
                       neuron_no]

"""
Define the training process of the QSNN from a random initialization.
"""


def training_for_sample(sa):
    """
    :param sa: the index of the initialization sample
    :return: the values of the loss and robustness of each iterations
    """
    global super_operator_unitary
    global ctrl_output
    global ctrl_unitary
    gama_out = gamma_out_for_for_sample[sa]
    
    # initialize the parameters in (2) Unitary evolution
    times_list_unitary = np.linspace(0, (t_unitary-1)/2, t_unitary)   # 输入时间 步长0.2
    parameters_list_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
    for i in range(0, int((N-3)*(N-4)/2)):
        for t in range(0, t_unitary):
            parameters_list_unitary[t, i] = h[i]  # 控制哈密顿算子连接
    adj_unitary = []   # 更新时只有哈密顿连接能够被改变
    for i in range(0, int((N-3)*(N-4)/2)):
        adj_unitary.append(i)
    ctrl_unitary = vll.Controller(times_list_unitary, parameters_list_unitary, time_invariant=time_indep_params_list, adjustable=adj_unitary)

    # initialize the parameters in (3) Output
    times_list_output = np.linspace(0, (t_output-1)/2, t_output)   # 输入时间 步长0.2
    parameters_list_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
    for t in range(0, t_output):
        for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
            parameters_list_output[t, i] = gama_out[i]
    adj_output = []
    for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):  # 更新时只有输出lindblad算子能够被改变
        adj_output.append(i)
    ctrl_output = vll.Controller(times_list_output, parameters_list_output, time_invariant=time_indep_params_list, adjustable=adj_output)

    """
    Calculate the loss, gradient, robustness in the case of the initial parameters
    """

    super_operator_unitary = get_super_operator_output(parameters_list_output)
    grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
    grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
    cost = 0
    robustness_output = 0
    result = []
    for ss in range(0, train_size):
        result.append(tot_solve(ss, parameters_list_unitary, parameters_list_output, get_robustness=True))
    for ii in result:
        cost += ii[0]/train_size
        grad_ave_unitary += ii[1]/train_size
        grad_ave_output += ii[2]/train_size
        robustness_output += ii[3]/train_size

    """
    Train
    """
    # define total number of iterations
    total_update_num = 100
    cost_list = [cost]
    robustness_output_list = [robustness_output]
    update_list = [0]
    update_num = 0

    while update_num < total_update_num:
        # define learning rate
        learning_rate_h_0 = 0.1
        learning_rate_gama_0 = 1
        UPDATE_NUM = 15
        learning_rate_gama = learning_rate_gama_0/(1+update_num/UPDATE_NUM)
        learning_rate_h = learning_rate_h_0/(1+update_num/UPDATE_NUM)

        # update the parameters
        parameters_list_unitary = parameters_list_unitary - learning_rate_h * grad_ave_unitary
        parameters_list_output = parameters_list_output - learning_rate_gama * grad_ave_output

        # calculate the loss, gradient, robustness in the case of the new parameters
        super_operator_unitary = get_super_operator_output(parameters_list_output)
        cost = 0
        robustness_output = 0
        grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
        grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
        result = []
        for ss in range(0, train_size):
            result.append(tot_solve(ss, parameters_list_unitary, parameters_list_output, get_robustness=True))
        for ii in result:
            cost += ii[0]/train_size
            grad_ave_unitary += ii[1]/train_size
            grad_ave_output += ii[2]/train_size
            robustness_output += ii[3]/train_size
        update_num += 1
        cost_list.append(cost)
        update_list.append(update_num)
        robustness_output_list.append(robustness_output)

    return cost_list, robustness_output_list

super_operator_2 = 0
ctrl_unitary = 0
ctrl_output = 0

pool = multiprocessing.Pool(processes=24)
result = []
for sample in range(0, sample_num):
    result.append(pool.apply_async(training_for_sample, (sample,)))
pool.close()
pool.join()
cost_list_for_sample_list = []
robustness_output_for_sample_list = []

for i in result:
    cost_list_for_sample_list.append(i.get()[0])
    robustness_output_for_sample_list.append(i.get()[1])

"""
Give the result of the training and test
"""

file_handle = open('sentence_recognition_loss_h0.1', mode='w')
file_handle.write(str(cost_list_for_sample_list))
file_handle = open('sentence_recognition_robustness_h0.1', mode='w')
file_handle.write(str(robustness_output_for_sample_list))






