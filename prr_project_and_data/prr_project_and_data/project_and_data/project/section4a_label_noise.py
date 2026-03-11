import numpy as np
import qutip as qt
import valedian211113 as vll
from qutip import *
import random
import multiprocessing
import copy
import string
import nltk
from nltk.corpus import stopwords
import nltk.stem
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
np.set_printoptions(threshold=np.inf)

"""
It is the code used to evaluate the label robustness of the QSNN.
The result is shown in section IV. A.
It is an example to train the QSNN with a specific initialization and specific error labels.
"""

"""
Set the training set
"""

x_train = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
            'sun gold day i', 'day nothing dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']

"""
Pre process the training data
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
    :param word_sequence_list: x_train 
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

encode_word_list = [('gold', 1), ('sun', 2), ('dawn', 3), ('love', 4), ('stay', 5), ('day', 6), ('go', 7), ('noth', 8)]
encode_word_dict = dict(encode_word_list)
N = len(encode_word_list)+3  # the total numbers of the neurons

"""
Initialize the network
"""

# Hamiltonian
h = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(0, int((N-3)*(N-4)/2)):
    h[i] = 0.1
# input Linblad operator
gamma_in = 1
# output Lindblad operator
gamma_out = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
    gamma_out[i] = 0.1
# all the parameters are time independent
time_indep_params_list = list(range(int((N-3)*(N-4)/2)+3*(N-3)))

# evolution time
t_input = 20
t_unitary = 20
t_output = 20
delta_t_for_s = [(int(t_input/words_in_se[s])) for s in range(0, train_size)]

# initialize the parameters in (2) Unitary evolution
times_list_unitary = np.linspace(0, (t_unitary-1)/2, t_unitary)
parameters_list_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
for i in range(0, int((N-3)*(N-4)/2)):
    for t in range(0, t_unitary):
        parameters_list_unitary[t, i] = h[i]
adj_unitary = list(range(0, int((N-3)*(N-4)/2)))  # parameters that can be adjusted
ctrl_unitary = vll.Controller(times_list_unitary, parameters_list_unitary, time_invariant=time_indep_params_list, adjustable=adj_unitary)

# initialize the parameters in (3) Output
times_list_output = np.linspace(0, (t_output-1)/2, t_output)
parameters_list_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
for t in range(0, t_output):
    for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
        parameters_list_output[t, i] = gamma_out[i]
adj_output = []
for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
    adj_output.append(i)
ctrl_output = vll.Controller(times_list_output, parameters_list_output, time_invariant=time_indep_params_list, adjustable=adj_output)


"""
Construct quantum stochastic neural networks
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
# get rho_in
rho_in_list = [qt.vector_to_operator(input_liouv_for_words_in(cleaned_text, s) * qt.operator_to_vector(rho_0)) for s in range(train_size)]

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
    """
    ctrl_output.safe_set('parameters', parameters_list)
    plc_constant = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    output_dm = plc_constant(input_dm)
    # training
    if get_grad:
        sr2 = vll.VariationalLearningPLC.LearningSubroutines(
            pH_pp, pC_pp_list, dl_df_output, loss=loss_output)
        vlplc = vll.VariationalLearningPLC(mdl, ctrl_output, sr2)
        return vlplc._gradient(input_dm, desired_output_dm), 1-((output_dm * desired_output_dm).tr())  # gradient, loss
    # test
    else:
        return (output_dm * desired_output_dm).tr()  # accuracy


def get_super_operator_output(param):
    """
    :return: the super operator of the evolution (3) output
    """
    ctrl_output.safe_set('parameters', param)
    plc = vll.ParameterizedLindbladChannel(mdl, ctrl_output)
    super_2 = plc.get_super_operator()
    return super_2


def tot_solve(s, parameters_list_u, parameters_list_o):
    """
    evolution of the training input and calculation of the gradient
    :param s: input sequence index
    :return: loss, the gradient of \vec{h} and \vec{gamma}
    """
    result_unitary_evolution_list = unitary_evolution(parameters_list_u, rho_in_list[s], desired_rho_out_list[s], get_grad=True)
    output_dm_list_unitary = result_unitary_evolution_list[1]
    grad_ave_u = result_unitary_evolution_list[0]
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_unitary, desired_rho_out_list[s], get_grad=True)
    c = result_output_evolution_list[1]
    grad_ave_o = result_output_evolution_list[0]
    return c, grad_ave_u, grad_ave_o


def label_error_tot_solve(s, parameters_list_u, parameters_list_o):
    """
    use corrected labels
    evolution of the training input and calculation of the gradient
    :param s: input sequence index
    :return: loss, the gradient of \vec{h} and \vec{gamma}
    """
    result_unitary_evolution_list = unitary_evolution(parameters_list_u, rho_in_list[s], corrected_desired_rho_out_list[s], get_grad=True)
    output_dm_list_unitary = result_unitary_evolution_list[1]
    grad_ave_u = result_unitary_evolution_list[0]
    result_output_evolution_list = output_evolution(parameters_list_o, output_dm_list_unitary, corrected_desired_rho_out_list[s], get_grad=True)
    c = result_output_evolution_list[1]
    grad_ave_o = result_output_evolution_list[0]
    return c, grad_ave_u, grad_ave_o


"""
Define labels (desired output density matrices) of the sequences
"""

neuron_yes = basis(N, N-1)*basis(N, N-1).dag()   # Yes
neuron_no = basis(N, N-2)*basis(N, N-2).dag()    # No
desired_rho_out_list = [neuron_yes, neuron_yes, neuron_no, neuron_no, neuron_yes, neuron_yes,
                neuron_no, neuron_no, neuron_yes, neuron_yes, neuron_no, neuron_no]           # labels with error

# the other 2 sets of error labels used in our simulation
# desired_rho_out_list = [neuron_no, neuron_no, neuron_yes, neuron_yes, neuron_yes, neuron_yes,
#                 neuron_yes, neuron_yes, neuron_no, neuron_no, neuron_no, neuron_no]         
# desired_output_list = [neuron_yes, neuron_yes, neuron_no, neuron_no, neuron_yes, neuron_yes,
#                        neuron_no, neuron_no, neuron_no, neuron_no, neuron_yes, neuron_yes]

corrected_desired_rho_out_list = [neuron_yes, neuron_yes, neuron_yes, neuron_yes, neuron_yes, neuron_yes, 
                          neuron_no, neuron_no, neuron_no, neuron_no, neuron_no, neuron_no]   # correct labels

"""
Calculate the loss and gradient in the case of the initial parameters
"""

super_operator_output = get_super_operator_output(parameters_list_output)
grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
cost = 0

pool = multiprocessing.Pool(processes=train_size)
result = []
for s in range(0, train_size):
    result.append(pool.apply_async(tot_solve, (s, parameters_list_unitary, parameters_list_output,)))
pool.close()
pool.join()
for i in result:
    cost += i.get()[0]/train_size
    grad_ave_unitary += i.get()[1]/train_size
    grad_ave_output += i.get()[2]/train_size

"""
Train
"""

cost_list = [cost]
update_list = [0]
update_num = 0
# define total number of iterations
total_update_num = 200
# correct labels after correct_step iterations 
correct_step = 100

while update_num < total_update_num:
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
    # update the parameters
    parameters_list_unitary = parameters_list_unitary - learning_rate_h * grad_ave_unitary
    parameters_list_output = parameters_list_output - learning_rate_gama * grad_ave_output
    
    # calculate the loss and gradient in the case of the new parameters
    super_operator_output = get_super_operator_output(parameters_list_output)
    grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
    cost = 0
    grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
    
    # train with error labels
    if update_num < correct_step or update_num == correct_step:
        pool = multiprocessing.Pool(processes=train_size)
        result = []
        for s in range(0, train_size):
            result.append(pool.apply_async(tot_solve, (s, parameters_list_unitary, parameters_list_output,)))
        pool.close()
        pool.join()
        for i in result:
            cost += i.get()[0]/train_size
            grad_ave_unitary += i.get()[1]/train_size
            grad_ave_output += i.get()[2]/train_size
        update_num = update_num + 1
        cost_list.append(cost)
        update_list.append(update_num)
    # train with correct labels
    else:
        pool = multiprocessing.Pool(processes=train_size)
        result = []
        for s in range(0, train_size):
            result.append(pool.apply_async(label_error_tot_solve, (s, parameters_list_unitary, parameters_list_output,)))
        pool.close()
        pool.join()
        for i in result:
            cost += i.get()[0]/train_size
            grad_ave_unitary += i.get()[1]/train_size
            grad_ave_output += i.get()[2]/train_size

        update_num = update_num + 1
        cost_list.append(cost)
        update_list.append(update_num)

"""
Give the output information of the training and test
"""

# loss
file_handle = open('label_error_loss', mode='w')
file_handle.write(str(cost_list))








