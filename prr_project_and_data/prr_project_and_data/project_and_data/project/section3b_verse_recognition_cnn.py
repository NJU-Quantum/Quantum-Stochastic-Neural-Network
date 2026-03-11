import numpy as np
import qutip as qt
import multiprocessing
import valedian211113 as vll
from qutip import *
import random
import copy
import math
import string
import nltk
from nltk.corpus import stopwords
import nltk.stem
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
np.set_printoptions(threshold=np.inf)

"""
It is the code used to train the classical neural network to complete the verse recognition task.
We construct the networks with the same structure as the QSNN, and optimize them with gradient descent.
The result is shown in section III. B.
We will process the word input sequences with QSNN to get the input data first, namely, the probabilities on the hidden layer of the QSNN is the input of the classical neural network.
"""

"""
Set the training set
"""

x_train = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
            'sun gold day i', 'day nothing dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']
x_test = ['so dawn goes down to day', 'nothing gold can stay', 'i love to stay here until the dawn', 'i love to go out for love']

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

test_cleaned_text = pre_process(x_test)
test_size = len(x_test)
test_words_in_se = [len(test_cleaned_text[s]) for s in range(0, test_size)]

"""
Encode the word into the neurons
"""

encode_word_list = [('gold', 1), ('sun', 2), ('dawn', 3), ('love', 4), ('stay', 5), ('day', 6), ('go', 7), ('noth', 8)]
encode_word_dict = dict(encode_word_list)
N = len(encode_word_list)+3  # the total numbers of the neurons

# hyper-parameters of the input evolution
gamma_in = 1
t_input = 20
delta_t_for_s = [(int(t_input/words_in_se[s])) for s in range(0, train_size)]
test_delta_t_for_s = [(int(t_input/test_words_in_se[s])) for s in range(0, test_size)]
time_indep_params_list = list(range(int((N-3)*(N-4)/2)+3*(N-3)))

"""
Construct quantum neural networks to get the input data
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
test_rho_in_list = [qt.vector_to_operator(input_liouv_for_words_in(test_cleaned_text, ts) * qt.operator_to_vector(rho_0)) for ts in range(test_size)]
# get the probability of the neuron in the hidden layers
neuron_hidden = [qt.basis(N, neuron) * qt.basis(N, neuron).dag() for neuron in range(1, N-3+1)]

"""
Classical Neuron Network
"""

"""
Input
"""

# data
input_data = [[(neuron_hidden[i]*rho_in_list[s]).tr() for i in range(0, N-3)] for s in range(train_size)]
test_input_data = [[(neuron_hidden[i]*test_rho_in_list[s]).tr() for i in range(0, N-3)] for s in range(test_size)]

# label
y_no_list = [6, 7, 8, 9, 10, 11]  # the index of the sequences labeled no
y_yes_list = [0, 1, 2, 3, 4, 5]   # the index of the sequences labeled yes

"""
Initialize of the parameters
"""

sample_num = 15
w_list_for_sample = []  # weights
b_no_for_sample = []    # bias
b_yes_for_sample = []   # bias
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
    """
    evolution of the training input and calculation of the gradient
    :param parameter_w: weights
    :param parameter_bno: bias
    :param parameter_byes: bias
    :return: loss and the gradient
    """
    # calculate loss
    w_no, w_yes = [], []
    for ii in range(0, 2*(N-3), 2):
        w_no.append(parameter_w[ii])
        w_yes.append(parameter_w[ii+1])

    z_no = [0 for s in range(0, train_size)]
    z_yes = [0 for s in range(0, train_size)]
    softmax_no = [0 for s in range(0, train_size)]
    softmax_yes = [0 for s in range(0, train_size)]

    for s in range(0, train_size):
        z_no[s] = sum(w * p for w, p in zip(w_no, input_data[s])) + parameter_bno
        z_yes[s] = sum(w * p for w, p in zip(w_yes, input_data[s])) + parameter_byes
        softmax_no[s] = 1/(1+math.exp(z_yes[s]-z_no[s]))
        softmax_yes[s] = 1/(1+math.exp(z_no[s]-z_yes[s]))

    no_sents = 0
    yes_sents = 0
    for ss in y_no_list:
        no_sents += softmax_no[ss]
    for ss in y_yes_list:
        yes_sents += softmax_yes[ss]
    cost = 1-(no_sents+yes_sents)/train_size  # loss

    # calculate the gradient
    pl_pw_list = []
    for w in range(0, N-3):
        pl_pw_no = 0
        pl_pbno = 0
        pl_pw_yes = 0
        pl_pbyes = 0
        for ss in range(0, train_size):
            if ss in y_no_list:
                pl_pw_no += softmax_no[ss]*(softmax_no[ss]-1)*input_data[ss][w]
                pl_pbno += softmax_no[ss]*(softmax_no[ss]-1)/train_size
            if ss in y_yes_list:
                pl_pw_yes += softmax_yes[ss]*(softmax_yes[ss]-1)*input_data[ss][w]
                pl_pbyes += softmax_yes[ss]*(softmax_yes[ss]-1)/train_size

        pl_pw_list.append(pl_pw_no/train_size)
        pl_pw_list.append(pl_pw_yes/train_size)

    return cost, pl_pw_list, pl_pbno, pl_pbyes


def test_evolution(parameter_w, parameter_bno, parameter_byes):
    """
    evolution of the test input
    :param parameter_w: weights
    :param parameter_bno: bias
    :param parameter_byes: bias
    :return: accuracy
    """
    w_no, w_yes = [], []
    for ii in range(0, 2*(N-3), 2):
        w_no.append(parameter_w[ii])
        w_yes.append(parameter_w[ii+1])
    z_no = [0 for s in range(0, test_size)]
    z_yes = [0 for s in range(0, test_size)]
    softmax_yes = [0 for s in range(0, test_size)]
    for s in range(0, test_size):
        z_no[s] = sum(w * p for w, p in zip(w_no, test_input_data[s])) + parameter_bno
        z_yes[s] = sum(w * p for w, p in zip(w_yes, test_input_data[s])) + parameter_byes
        softmax_yes[s] = 1/(1+math.exp(z_no[s]-z_yes[s]))
    return softmax_yes  # accuracy


def training_test_for_sample(sam):
    """
    training process beginning with a initialization sample and evaluation with test set after each iteration
    :param sam: initialization sample index
    :return: loss and accuracy
    """
    # define total number of iterations
    total_update_num = 5

    # calculate the loss and gradient in the case of the initial parameters
    w_list = w_list_for_sample[sam]
    b_no = b_no_for_sample[sam]
    b_yes = b_yes_for_sample[sam]
    result_for_sample = evolution(w_list, b_no, b_yes)
    loss_list = [result_for_sample[0]]
    pl_pw = result_for_sample[1]
    pl_pb_no = result_for_sample[2]
    pl_pb_yes = result_for_sample[3]

    # use test sets to evaluate the initial model
    test_result_for_sample = test_evolution(w_list, b_no, b_yes)
    accuracy_for_sent1_for_sample = [test_result_for_sample[0]]
    accuracy_for_sent2_for_sample = [test_result_for_sample[1]]
    accuracy_for_sent3_for_sample = [test_result_for_sample[2]]
    accuracy_for_sent4_for_sample = [test_result_for_sample[3]]

    # iteration
    update_num = 0
    while update_num < total_update_num:
        # define the learning rate
        rate_w_0 = 3
        rate_b_0 = 3
        UPDATE_NUM = 15
        if update_num < 100 or update_num == 100:
            rate_w = rate_w_0/(1+update_num/UPDATE_NUM)
            rate_b = rate_b_0/(1+update_num/UPDATE_NUM)
        else:
            rate_w = rate_w_0/(1+100/UPDATE_NUM)
            rate_b = rate_b_0/(1+100/UPDATE_NUM)

        # update the parameters
        for ii in range(0, 2*(N-3)):
            w_list[ii] = w_list[ii] - rate_w * pl_pw[ii]
        b_no = b_no - rate_b * pl_pb_no
        b_yes = b_yes - rate_b * pl_pb_yes

        # calculate the loss and gradient in the case of the new parameters
        result_for_sample = evolution(w_list, b_no, b_yes)
        loss = result_for_sample[0]
        loss_list.append(loss)
        pl_pw = result_for_sample[1]
        pl_pb_no = result_for_sample[2]
        pl_pb_yes = result_for_sample[3]

        # use test sets to evaluate the initial model in the case of the new parameters
        test_result_for_sample = test_evolution(w_list, b_no, b_yes)
        accuracy_for_sent1_for_sample.append(test_result_for_sample[0])
        accuracy_for_sent2_for_sample.append(test_result_for_sample[1])
        accuracy_for_sent3_for_sample.append(test_result_for_sample[2])
        accuracy_for_sent4_for_sample.append(test_result_for_sample[3])

        update_num = update_num + 1
    return loss_list, accuracy_for_sent1_for_sample, accuracy_for_sent2_for_sample, accuracy_for_sent3_for_sample, accuracy_for_sent4_for_sample

"""
Train and test with all initialization samples
"""

result = []
for sample in range(0, sample_num):
    result.append(training_test_for_sample(sample))
loss_list_for_sample_list = []
accuracy_for_sent1_for_sample_list = []
accuracy_for_sent2_for_sample_list = []
accuracy_for_sent3_for_sample_list = []
accuracy_for_sent4_for_sample_list = []
for i in result:
    loss_list_for_sample_list.append(i[0])
    accuracy_for_sent1_for_sample_list.append(i[1])
    accuracy_for_sent2_for_sample_list.append(i[2])
    accuracy_for_sent3_for_sample_list.append(i[3])
    accuracy_for_sent4_for_sample_list.append(i[4])

"""
Give the result of the training and test
"""

# loss
Loss = open('verse_recognition_classicalNN_loss.txt', mode='w')
Loss.write(str(loss_list_for_sample_list))
# accuracy
sent1 = open('verse_recognition_classicalNN_test1.txt', mode='w')
sent1.write(str(accuracy_for_sent1_for_sample_list))
sent2 = open('verse_recognition_classicalNN_test2.txt', mode='w')
sent2.write(str(accuracy_for_sent2_for_sample_list))
sent3 = open('verse_recognition_classicalNN_test3.txt', mode='w')
sent3.write(str(accuracy_for_sent3_for_sample_list))
sent4 = open('verse_recognition_classicalNN_test4.txt', mode='w')
sent4.write(str(accuracy_for_sent4_for_sample_list))





