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
    h[i] = 0.1

gama = 1

# 输出随机初始化
# gama_out_sample = []
# for sample in range(0, 15):
#     gama_out = [0 for i in range(0, int((N-3)*(N-4)/2+3*(N-3)))]
#     for i in range(int((N-3)*(N-4)/2+N-3), int((N-3)*(N-4)/2)+3*(N-3)):
#         gama_out[i] = random.uniform(-1, 1)
#     gama_out_sample.append(gama_out)
# print(gama_out_sample)

# 0-29样本
gama_out_sample = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.35261623193513825, 0.6368055964900352, -0.4644137884382251, 0.517599574417537, -0.9527636233435597, 0.6332846644046495, -0.6821379048031151, 0.3275074012285355, 0.8301162662877188, 0.4018847870522193, -0.9366832411892216, -0.894892847767691, 0.4665280985659208, 0.5162576243894672, -0.7790141019907124, 0.3642387881571292], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4372681079629521, -0.8257744974944066, -0.07467304986806478, -0.3921664348724998, 0.6121064679684092, 0.24436218938390675, 0.3141194047131508, 0.06975203948918662, -0.5328876867062289, 0.3509399946427303, 0.1847325151957575, 0.7831311611719054, 0.21733693662800735, 0.21368335642026715, 0.06929330430107439, 0.6966796636254327], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2877924774024745, -0.7743761593389564, -0.2249308143901676, 0.772395936027273, -0.4923218068028421, 0.7266068504280807, 0.9935716951695761, 0.3506157540734933, 0.43810053509588753, 0.22179700410404135, -0.48742390689742465, -0.936111498904123, 0.44566828245402323, 0.3165659303791084, 0.40712526880240096, 0.9059107414525751], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3396321018731929, 0.6112969795287209, 0.2707561250657047, -0.029525425889412205, 0.3630907271276771, -0.7783294362572626, -0.22937298770474013, -0.49407388078162096, 0.7794586137770225, -0.009078141100703263, -0.8601001272842077, 0.5524938659394292, 0.20058544626266528, 0.9546424219366216, -0.40100266754470026, 0.8450112494253441], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.678198338545126, 0.9385210545173803, -0.7150925598232267, -0.3563733922648351, 0.6459588642104526, -0.46529346974104624, -0.607482604432126, 0.17851759242241894, -0.31123457162476464, -0.6608942275367518, -0.9379161128739166, 0.10686694644240058, -0.2627284263778269, 0.7583565491634994, -0.1971234882023949, -0.36810494848713615], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.2532773509375492, 0.0987875683635202, -0.7335791179557727, -0.4457544083370082, 0.810486276581929, -0.11162982046466152, 0.10101527834932367, -0.5341525083136802, 0.018465755926601712, 0.8672219913346664, 0.8802951545204003, -0.18785308671743217, -0.17865987322359866, -0.08391925601651562, 0.4104915589813174, 0.05303100521695203], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.36530441503156696, 0.3351976151855278, 0.872243042095469, 0.9873891622417528, 0.887349468725448, 0.42070179497001825, -0.8735116646284411, 0.934150587443322, 0.41100334925706794, -0.9376040760400672, -0.9339999326082242, -0.8296371761781098, -0.807122195120656, 0.011746728812152663, -0.12567374113707563, -0.43295604819944145], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.9048040188924664, -0.27293901586202196, -0.13199291788948964, 0.6153068042838505, 0.7360988662415717, -0.03511921628931747, -0.190744460290857, -0.8852460041461592, -0.1474611994005861, 0.6538694908437148, 0.9060069960820292, 0.14016808146965776, 0.09887718268564094, 0.270140745504619, -0.12543319174015655, 0.36227019682695394], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.40219470403068613, 0.37889754096605777, 0.9729270986036591, 0.8289218781569194, -0.28661462037733654, -0.6428270361117032, 0.1524669282206077, -0.029200022156083083, -0.8369776748458313, 0.36736479117597143, -0.907256321566901, -0.004021823424446458, -0.32570009383187526, -0.7669313677386023, 0.1692946078486639, -0.13615619704042792], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8250778007666795, -0.8227573034854618, 0.20716624244554938, -0.5271965854110139, -0.605792525011632, 0.48016383500804305, -0.3343120930964596, -0.11907947571863309, -0.8431177342925442, -0.2542278798116573, 0.1618402954268925, -0.4109516213225417, 0.1124911109839013, 0.8091409462329329, -0.08219788920867188, 0.24335767223594806], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.18422276481582167, 0.7915993320142389, -0.41934415990052987, -0.7575279173298457, -0.8926109102252786, -0.36694932674509273, -0.6740991702835233, 0.858343722554384, -0.5006211339779478, 0.8156064457438967, -0.3895852167812297, 0.6090119425714657, 0.6082291357014569, 0.8871575558227198, -0.929723744068607, -0.057241765564114866], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.49860112738032814, 0.5127594669061619, -0.9219258394726104, 0.4610811259092109, 0.43021141584844336, -0.367525301029179, -0.2546332542241223, -0.26500442448408656, -0.347982258096037, -0.5753271365263635, 0.9144164147193876, -0.3296074685255943, -0.14970401995041493, 0.09585981321086567, 0.5192132611038873, -0.2499847477870074], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.41834759435814983, 0.6517519252913837, -0.054182398495470574, 0.14828623874381175, 0.10447186267552167, 0.6448293878404341, 0.337661921879727, 0.17063200314809657, 0.3955754250668737, -0.6299916946450592, -0.5038813031692988, 0.23207892209845649, -0.35443186866492504, -0.9098646305722211, 0.5915620068479988, 0.5479489289484063], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.42069775316419356, 0.28064771514017406, 0.7666147643627419, 0.9317028933190588, 0.8433282732964849, 0.6818213650783678, 0.2052703270406917, -0.22291344583982053, -0.9493435308454836, 0.9679263110506118, 0.29862009092296304, 0.8568609404225427, -0.009120135542552976, -0.42751431397816564, 0.8414497384699589, 0.8197005950826493], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8738418673657959, 0.9701731112262402, 0.46052630426660945, 0.4276745787452909, -0.14355047088263162, -0.5614063029870453, -0.8641551856434981, -0.6026431255299052, -0.06843597718406214, 0.17676260454832127, 0.13321429744229274, -0.4775674080573167, -0.4770887793398575, -0.5311180043225805, -0.29995665390835957, 0.15343979630724114], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3198289686772793, -0.9386589890195829, -0.7309717349616891, 0.9219486102248766, 0.08715983943608752, 0.7106145788549698, -0.19890989764696543, -0.3953300881120505, 0.3684588621457703, -0.6991052491478285, -0.31152419517272434, 0.13188806392400498, -0.7196220898556143, 0.12169708821391345, -0.7197315308622758, 0.03841018211652747], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.14214001825834832, 0.14519174266598678, 0.8384698640679029, -0.8091152216868647, -0.8106669294773103, 0.29417391807940074, -0.310733722372333, 0.04796671509532113, -0.8348747374731114, 0.4156242338600413, 0.7277647327494892, 0.2534326113456835, -0.44112620897063803, 0.6228495639292468, -0.7494112015815575, 0.7308907536786198], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.45768459222203006, 0.8155329160500502, 0.731690402410387, 0.9003437840853821, -0.29940481399375884, -0.33874379858992953, 0.3462199668528474, 0.002440000244203766, -0.1613618749483534, 0.6110419614419449, -0.5854870371530672, 0.07256797440173646, -0.36533304191857585, 0.5969838342473079, -0.2909872252312524, -0.1994371292582524], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.09592749558014924, 0.818466702689411, 0.31647982102977745, -0.043254239166948816, -0.8471373900327066, -0.8685747936528714, -0.5290199704692982, -0.41086614965366564, 0.3283630156879942, 0.7682034360959984, 0.579955831352023, -0.6580978041266865, 0.7014196783564777, -0.7495864113471615, -0.9280367614218574, -0.08569851289994079], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.6660523115338453, 0.41814849049210556, -0.0651013186585998, 0.3064993606646256, -0.3129929863766374, 0.5869351068236055, -0.892383545415087, 0.7160389548544237, -0.42019631427374304, -0.818709757280458, 0.1178654031908899, -0.15289046664401162, 0.7779944859031522, 0.3182654546930803, 0.8391718046543444, -0.7400245572732695], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.13918310841525372, 0.6681785710393495, 0.3487057386015513, 0.9266889694531273, 0.4506016984300336, -0.8389292073739665, -0.22320836765614493, -0.42811681535169077, 0.9882383804910666, -0.1255621578160786, 0.5826488722559, -0.5788598997406382, -0.6369236756938943, -0.6148424226424991, -0.38738268001495824, 0.03357373166672417], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1544060095825952, 0.9910087372807033, 0.26453773940508385, 0.850209630023165, -0.25042747693774037, 0.7445849226588548, -0.046540558901105644, 0.23365657515292626, 0.005126637241177567, 0.1077859673782886, 0.006911792248099458, 0.7498481823128702, -0.3596153774617219, -0.6036125830512489, 0.39993290375626067, -0.5019337832068052], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8449535011865472, 0.678449658441975, -0.6242287415662358, -0.5380436642292719, -0.6460856408752735, 0.7762160539037806, -0.03322132895734997, -0.6295556009364138, -0.22781694406097852, -0.13597300542145874, 0.012593755885451863, -0.4083235379586676, -0.8691902517130436, -0.4767185056118117, 0.4730544315917402, 0.2740896682171845], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.8321491769868317, 0.6269280164241948, -0.3082862104754165, -0.904764177161286, 0.852658363327707, 0.0895865858443825, -0.2638930694181192, -0.941408939901416, -0.6112336403568339, -0.39300022439161286, -0.997247401116274, 0.39762870087506075, 0.3254044561256495, -0.6106609136702694, -0.09865771618024732, -0.029697487297142278], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5525801027397661, 0.9998865204648304, 0.3110029070635705, -0.42507805395396026, -0.4451651835842223, 0.954729451685622, 0.6926872760639011, -0.798401635366887, -0.3840421853447702, -0.3448600973666036, 0.7828726047994699, -0.5886394410635034, 0.20744606139522181, 0.6005792496442315, 0.890713789476512, 0.8616009855325888], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.16759904091964106, 0.5154640463338567, -0.7395785457164799, -0.5657032020054102, -0.821384944040028, 0.7168064388870374, 0.2741617090828439, -0.9143017926983494, -0.8965869812183269, 0.1389120830369306, -0.298594362605886, 0.2690378947004657, -0.4098268769580702, -0.7386992143040112, 0.56838358937753, -0.051938920296027646], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.9749844169548165, 0.5203513081281419, 0.7194900876334802, -0.8747177057227504, -0.1510605398484539, -0.2443736471879614, 0.2692319244375754, 0.986455042660564, -0.28166311390131105, -0.12137362414316666, -0.35829542638237943, 0.7119679268878476, 0.45255958074149394, 0.4104995381859782, -0.8199263611828209, 0.6961406595400648], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.7759122472741753, -0.10810237041100468, 0.4235138701099719, 0.927619716192666, -0.8604889173142829, 0.5276376380794199, -0.37778416877333654, -0.16376426516297138, -0.04715105890072624, -0.7125383601632336, 0.23247670289880862, 0.7703041613700414, -0.18638290131629676, -0.6192550252871738, -0.3415903545591037, 0.23087539877805763], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.6328912542629572, -0.6848382565145783, -0.9642889992867292, -0.38273860466863496, -0.3883604617509997, 0.41135797625553927, -0.46410347251930495, 0.7858139134315261, -0.3644865281258365, 0.9082685851641155, 0.671369527113735, 0.9242903309916355, 0.9259398985199787, 0.9417667266016208, 0.2436168651116435, -0.21192614136126653]]
gama_out = gama_out_sample[23]


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


file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr23-h0.1-thtg-loss', mode='w')
file_handle.write(str(cost_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr23-h0.1-thtg-sent1', mode='w')
file_handle.write(str(test_cost_sent1_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr23-h0.1-thtg-sent2', mode='w')
file_handle.write(str(test_cost_sent2_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr23-h0.1-thtg-sent3', mode='w')
file_handle.write(str(test_cost_sent3_list))
file_handle = open('/share/home/sjwu/wlj/project/3-layers/random-g/data/gr23-h0.1-thtg-sent4', mode='w')
file_handle.write(str(test_cost_sent4_list))









