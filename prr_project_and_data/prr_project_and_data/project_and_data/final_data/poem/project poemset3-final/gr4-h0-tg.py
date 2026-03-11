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


# language_sent = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
#                  'sun gold day i', 'nothing go dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']
# language_sent = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
#                  'sun gold day i', 'go nothing dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']  # newset2
language_sent = ['There is a gold sun at dawn', 'I love stay all day in the sun', 'He went for gold that day', 'He love gold but have nothing ', 'He loves the dawn of a day', 'I love the lovely sun',
                 'sun gold day i', 'day nothing dawn', 'gold goes love', 'stay love go sun', 'day gold nothing', 'stay dawn love go']  # newset3

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
    tagged_sent[i] = pos_tag(tokens[i])
    wnl = WordNetLemmatizer()
    lemmas_sent[i] = []
    for tag in tagged_sent[i]:
        wordnet_pos[i] = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent[i].append(wnl.lemmatize(tag[0], pos=wordnet_pos[i]))  # 词形还原

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
t_unitary = 20
t_output = 20


# 哈密顿量连接系数
h = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
for i in range(0, int((N-3)*(N-4)/2)):
    h[i] = 0

gama = 1

# 输出随机初始化
# gama_out_sample = []
# for sample in range(0, 20):
#     gama_out = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
#     for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
#         gama_out[i] = random.uniform(-1, 1)
#     gama_out_sample.append(gama_out)
# print(gama_out_sample)
gama_out_sample = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.35261623193513825, 0.6368055964900352, -0.4644137884382251, 0.517599574417537, -0.9527636233435597, 0.6332846644046495, -0.6821379048031151, 0.3275074012285355, 0.8301162662877188, 0.4018847870522193, -0.9366832411892216, -0.894892847767691, 0.4665280985659208, 0.5162576243894672, -0.7790141019907124, 0.3642387881571292], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4372681079629521, -0.8257744974944066, -0.07467304986806478, -0.3921664348724998, 0.6121064679684092, 0.24436218938390675, 0.3141194047131508, 0.06975203948918662, -0.5328876867062289, 0.3509399946427303, 0.1847325151957575, 0.7831311611719054, 0.21733693662800735, 0.21368335642026715, 0.06929330430107439, 0.6966796636254327], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2877924774024745, -0.7743761593389564, -0.2249308143901676, 0.772395936027273, -0.4923218068028421, 0.7266068504280807, 0.9935716951695761, 0.3506157540734933, 0.43810053509588753, 0.22179700410404135, -0.48742390689742465, -0.936111498904123, 0.44566828245402323, 0.3165659303791084, 0.40712526880240096, 0.9059107414525751], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3396321018731929, 0.6112969795287209, 0.2707561250657047, -0.029525425889412205, 0.3630907271276771, -0.7783294362572626, -0.22937298770474013, -0.49407388078162096, 0.7794586137770225, -0.009078141100703263, -0.8601001272842077, 0.5524938659394292, 0.20058544626266528, 0.9546424219366216, -0.40100266754470026, 0.8450112494253441], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.678198338545126, 0.9385210545173803, -0.7150925598232267, -0.3563733922648351, 0.6459588642104526, -0.46529346974104624, -0.607482604432126, 0.17851759242241894, -0.31123457162476464, -0.6608942275367518, -0.9379161128739166, 0.10686694644240058, -0.2627284263778269, 0.7583565491634994, -0.1971234882023949, -0.36810494848713615], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2532773509375492, 0.0987875683635202, -0.7335791179557727, -0.4457544083370082, 0.810486276581929, -0.11162982046466152, 0.10101527834932367, -0.5341525083136802, 0.018465755926601712, 0.8672219913346664, 0.8802951545204003, -0.18785308671743217, -0.17865987322359866, -0.08391925601651562, 0.4104915589813174, 0.05303100521695203], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.36530441503156696, 0.3351976151855278, 0.872243042095469, 0.9873891622417528, 0.887349468725448, 0.42070179497001825, -0.8735116646284411, 0.934150587443322, 0.41100334925706794, -0.9376040760400672, -0.9339999326082242, -0.8296371761781098, -0.807122195120656, 0.011746728812152663, -0.12567374113707563, -0.43295604819944145], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.9048040188924664, -0.27293901586202196, -0.13199291788948964, 0.6153068042838505, 0.7360988662415717, -0.03511921628931747, -0.190744460290857, -0.8852460041461592, -0.1474611994005861, 0.6538694908437148, 0.9060069960820292, 0.14016808146965776, 0.09887718268564094, 0.270140745504619, -0.12543319174015655, 0.36227019682695394], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.40219470403068613, 0.37889754096605777, 0.9729270986036591, 0.8289218781569194, -0.28661462037733654, -0.6428270361117032, 0.1524669282206077, -0.029200022156083083, -0.8369776748458313, 0.36736479117597143, -0.907256321566901, -0.004021823424446458, -0.32570009383187526, -0.7669313677386023, 0.1692946078486639, -0.13615619704042792], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8250778007666795, -0.8227573034854618, 0.20716624244554938, -0.5271965854110139, -0.605792525011632, 0.48016383500804305, -0.3343120930964596, -0.11907947571863309, -0.8431177342925442, -0.2542278798116573, 0.1618402954268925, -0.4109516213225417, 0.1124911109839013, 0.8091409462329329, -0.08219788920867188, 0.24335767223594806], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.18422276481582167, 0.7915993320142389, -0.41934415990052987, -0.7575279173298457, -0.8926109102252786, -0.36694932674509273, -0.6740991702835233, 0.858343722554384, -0.5006211339779478, 0.8156064457438967, -0.3895852167812297, 0.6090119425714657, 0.6082291357014569, 0.8871575558227198, -0.929723744068607, -0.057241765564114866], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.49860112738032814, 0.5127594669061619, -0.9219258394726104, 0.4610811259092109, 0.43021141584844336, -0.367525301029179, -0.2546332542241223, -0.26500442448408656, -0.347982258096037, -0.5753271365263635, 0.9144164147193876, -0.3296074685255943, -0.14970401995041493, 0.09585981321086567, 0.5192132611038873, -0.2499847477870074], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.41834759435814983, 0.6517519252913837, -0.054182398495470574, 0.14828623874381175, 0.10447186267552167, 0.6448293878404341, 0.337661921879727, 0.17063200314809657, 0.3955754250668737, -0.6299916946450592, -0.5038813031692988, 0.23207892209845649, -0.35443186866492504, -0.9098646305722211, 0.5915620068479988, 0.5479489289484063], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.42069775316419356, 0.28064771514017406, 0.7666147643627419, 0.9317028933190588, 0.8433282732964849, 0.6818213650783678, 0.2052703270406917, -0.22291344583982053, -0.9493435308454836, 0.9679263110506118, 0.29862009092296304, 0.8568609404225427, -0.009120135542552976, -0.42751431397816564, 0.8414497384699589, 0.8197005950826493], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1206522045173688, -0.09167836210203828, 0.260210626247926, -0.3861781982165138, 0.8306031038253907, 0.03828600583504782, 0.08467827490231872, -0.629894726244284, 0.7348234577927899, 0.2822456593879237, 0.038154747195275895, 0.1256664542539203, 0.7398257277775573, -0.8116075839888, 0.14381036102487932, 0.9316539806618203], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.38293550865314585, -0.212867473026209, -0.13910129469229404, -0.8800733505438103, 0.8235629151438368, 0.440011447025501, 0.6060664821294097, -0.9862784839679808, 0.08221729005273581, -0.13247885247356805, -0.9119214518562497, 0.815640354927833, 0.7427872321486291, -0.018595996967712525, 0.1581164678091571, 0.5158240134100307], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9354461168622785, 0.13220860334351925, 0.1431588720906254, 0.5420657700715823, 0.6066340449791621, 0.023162530230476097, 0.8516837442854004, 0.23566597976042702, 0.5248811889468301, 0.05015818106966852, 0.34735115506007874, 0.08838998902079354, 0.761898356764648, -0.7759199130856904, 0.43689114686822017, 0.5277566823324624], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3295530838737386, -0.3270956055798957, 0.8401228261007048, -0.8165400173687185, -0.9923913292648388, 0.5301193347620272, -0.29561430811802314, 0.06515725345537415, -0.7881059736384157, 0.1447787603290862, 0.2591349958842035, 0.08599453695726478, 0.5332922534414364, -0.6562401493436456, 0.5554800278444267, 0.8697848728164317], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2451082421945936, 0.2226994374199549, 0.27482185342614995, 0.23151966960959425, -0.30626918641230616, 0.7944529058864505, -0.5729620273657483, 0.4743935704495057, -0.4942752032730686, -0.4741311785011313, -0.39089497053749844, 0.21142928434890562, -0.06819926113819341, -0.1558589082219004, -0.7822401780692594, -0.12228085341510075], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8914061040980903, 0.4678755695533199, -0.04882633096343536, 0.9523145859100186, -0.7760174917446234, -0.4127043101618153, 0.9267878681781645, 0.8367088443820112, -0.5677870102908849, 0.3796840147145937, 0.10105642625430744, 0.19176832866310667, 0.15934409655767512, 0.6981490271364501, 0.04654606810184059, -0.967794295251778]]
gama_out = gama_out_sample[4]


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
        return (output_dm * desired_output_dm).tr()


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


def test_tot_solve(s, parameters_list_u, parameters_list_o):
    test_output_dm_list_unitary = unitary_evolution(parameters_list_u, test_output_dm_list_input[s], test_desired_output_list1[s], get_grad=False)
    test_c = output_evolution(parameters_list_o, test_output_dm_list_unitary, test_desired_output_list1[s], get_grad=False)
    return test_c   # 返回准确率


input_dm_input = qt.basis(N, 0)*qt.basis(N, 0).dag()
test_output_dm_list_input = []
tS_list = []
for ts in range(0, len(test_sent)):
    tS_list.append(test_input_liouv_for_words_in(ts))
    test_output_dm_list_input.append(qt.vector_to_operator(tS_list[-1] * qt.operator_to_vector(input_dm_input)))
# print(test_output_dm_list_input[0])


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
desired_output_list = [measure1, measure1, measure1, measure1, measure1, measure1,
                       measure0, measure0, measure0, measure0, measure0, measure0]
test_desired_output_list1 = [basis(N, N-1)*basis(N, N-1).dag() for i in range(0, len(test_words_in_sent))]


# 计算第一个梯度和代价
super_operator_2 = get_super_operator_2(parameters_list_output)
grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
cost = 0
grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))
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

# 计算测试集第一个代价
test_cost_average = 0
test_cost_list = []
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=test_input_num)
    test_result = []
    for ts in range(0, test_sents):
        test_result.append(pool.apply_async(test_tot_solve, (ts, parameters_list_unitary, parameters_list_output,)))
    pool.close()
    pool.join()
    for i in test_result:
        test_cost_average += i.get()/test_input_num
        test_cost_list.append(i.get())


cost_list = [cost]
update_list = [0]
update_num = 0

test_cost_average_list = [test_cost_average]
test_cost_sent1_list = [test_cost_list[0]]
test_cost_sent2_list = [test_cost_list[1]]
test_cost_sent3_list = [test_cost_list[2]]
test_cost_sent4_list = [test_cost_list[3]]

while update_num < 200:
    learning_rate_h_0 = 0.5
    learning_rate_gama_0 = 3
    UPDATE_NUM = 15
    if update_num < 100:
        learning_rate_gama = learning_rate_gama_0/(1+update_num/UPDATE_NUM)
        learning_rate_h = learning_rate_h_0/(1+update_num/UPDATE_NUM)
    else:
        learning_rate_gama = learning_rate_gama_0/(1+100/UPDATE_NUM)
        learning_rate_h = learning_rate_h_0/(1+100/UPDATE_NUM)

    parameters_list_unitary = parameters_list_unitary - learning_rate_h * grad_ave_unitary
    parameters_list_output = parameters_list_output - learning_rate_gama * grad_ave_output
    super_operator_2 = get_super_operator_2(parameters_list_output)

    grad_ave_unitary = np.zeros((len(times_list_unitary), int((N-3)*(N-4)/2)+3*(N-3)))
    cost = 0
    grad_ave_output = np.zeros((len(times_list_output), int((N-3)*(N-4)/2)+3*(N-3)))

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
        update_num = update_num + 1
        cost_list.append(cost)
        update_list.append(update_num)

    test_parameters_list_unitary = copy.deepcopy(parameters_list_unitary)
    test_parameters_list_output = copy.deepcopy(parameters_list_output)
    test_cost_average = 0
    test_cost_list = []
    if __name__ == "__main__":
        pool = multiprocessing.Pool(processes=test_input_num)
        test_result = []
        for ts in range(0, test_sents):
            test_result.append(pool.apply_async(test_tot_solve, (ts, test_parameters_list_unitary, test_parameters_list_output,)))
        pool.close()
        pool.join()
        for i in test_result:
            test_cost_average += i.get()/test_input_num
            test_cost_list.append(i.get())
        test_cost_average_list.append(test_cost_average)
        test_cost_sent1_list.append(test_cost_list[0])
        test_cost_sent2_list.append(test_cost_list[1])
        test_cost_sent3_list.append(test_cost_list[2])
        test_cost_sent4_list.append(test_cost_list[3])


file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr4-h0-tg-loss', mode='w')
file_handle.write(str(cost_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr4-h0-tg-sent1', mode='w')
file_handle.write(str(test_cost_sent1_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr4-h0-tg-sent2', mode='w')
file_handle.write(str(test_cost_sent2_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr4-h0-tg-sent3', mode='w')
file_handle.write(str(test_cost_sent3_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr4-h0-tg-sent4', mode='w')
file_handle.write(str(test_cost_sent4_list))









