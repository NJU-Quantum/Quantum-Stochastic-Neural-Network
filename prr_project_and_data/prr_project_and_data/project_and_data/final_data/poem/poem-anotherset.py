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
import data

classical_loss = open('D:\PyCharm5.0.3\project\classicalNN-poem19\classicalNN_loss.txt', 'r')
classical_sent1 = open('D:\PyCharm5.0.3\project\classicalNN-poem19\classicalNN_test1.txt', 'r')
classical_sent2 = open('D:\PyCharm5.0.3\project\classicalNN-poem19\classicalNN_test2.txt', 'r')
classical_sent3 = open('D:\PyCharm5.0.3\project\classicalNN-poem19\classicalNN_test3.txt', 'r')
classical_sent4 = open('D:\PyCharm5.0.3\project\classicalNN-poem19\classicalNN_test4.txt', 'r')

gr0_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr0-h0.1-thtg-loss', 'r')
gr0_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr0-h0.1-thtg-sent1', 'r')
gr0_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr0-h0.1-thtg-sent2', 'r')
gr0_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr0-h0.1-thtg-sent3', 'r')
gr0_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr0-h0.1-thtg-sent4', 'r')
gr0_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr0-h0-tg-loss', 'r')
gr0_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr0-h0-tg-sent1', 'r')
gr0_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr0-h0-tg-sent2', 'r')
gr0_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr0-h0-tg-sent3', 'r')
gr0_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr0-h0-tg-sent4', 'r')

gr1_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr1-h0.1-thtg-loss', 'r')
gr1_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr1-h0.1-thtg-sent1', 'r')
gr1_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr1-h0.1-thtg-sent2', 'r')
gr1_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr1-h0.1-thtg-sent3', 'r')
gr1_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr1-h0.1-thtg-sent4', 'r')
gr1_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr1-h0-tg-loss', 'r')
gr1_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr1-h0-tg-sent1', 'r')
gr1_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr1-h0-tg-sent2', 'r')
gr1_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr1-h0-tg-sent3', 'r')
gr1_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr1-h0-tg-sent4', 'r')

gr2_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr2-h0.1-thtg-loss', 'r')
gr2_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr2-h0.1-thtg-sent1', 'r')
gr2_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr2-h0.1-thtg-sent2', 'r')
gr2_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr2-h0.1-thtg-sent3', 'r')
gr2_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr2-h0.1-thtg-sent4', 'r')
gr2_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr2-h0-tg-loss', 'r')
gr2_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr2-h0-tg-sent1', 'r')
gr2_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr2-h0-tg-sent2', 'r')
gr2_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr2-h0-tg-sent3', 'r')
gr2_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr2-h0-tg-sent4', 'r')

gr3_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr3-h0.1-thtg-loss', 'r')
gr3_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr3-h0.1-thtg-sent1', 'r')
gr3_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr3-h0.1-thtg-sent2', 'r')
gr3_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr3-h0.1-thtg-sent3', 'r')
gr3_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr3-h0.1-thtg-sent4', 'r')
gr3_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr3-h0-tg-loss', 'r')
gr3_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr3-h0-tg-sent1', 'r')
gr3_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr3-h0-tg-sent2', 'r')
gr3_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr3-h0-tg-sent3', 'r')
gr3_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr3-h0-tg-sent4', 'r')

gr4_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr4-h0.1-thtg-loss', 'r')
gr4_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr4-h0.1-thtg-sent1', 'r')
gr4_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr4-h0.1-thtg-sent2', 'r')
gr4_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr4-h0.1-thtg-sent3', 'r')
gr4_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr4-h0.1-thtg-sent4', 'r')
gr4_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr4-h0-tg-loss', 'r')
gr4_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr4-h0-tg-sent1', 'r')
gr4_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr4-h0-tg-sent2', 'r')
gr4_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr4-h0-tg-sent3', 'r')
gr4_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr4-h0-tg-sent4', 'r')

gr5_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr5-h0.1-thtg-loss', 'r')
gr5_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr5-h0.1-thtg-sent1', 'r')
gr5_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr5-h0.1-thtg-sent2', 'r')
gr5_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr5-h0.1-thtg-sent3', 'r')
gr5_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr5-h0.1-thtg-sent4', 'r')
gr5_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr5-h0-tg-loss', 'r')
gr5_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr5-h0-tg-sent1', 'r')
gr5_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr5-h0-tg-sent2', 'r')
gr5_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr5-h0-tg-sent3', 'r')
gr5_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr5-h0-tg-sent4', 'r')

gr6_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr6-h0.1-thtg-loss', 'r')
gr6_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr6-h0.1-thtg-sent1', 'r')
gr6_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr6-h0.1-thtg-sent2', 'r')
gr6_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr6-h0.1-thtg-sent3', 'r')
gr6_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr6-h0.1-thtg-sent4', 'r')
gr6_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr6-h0-tg-loss', 'r')
gr6_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr6-h0-tg-sent1', 'r')
gr6_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr6-h0-tg-sent2', 'r')
gr6_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr6-h0-tg-sent3', 'r')
gr6_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr6-h0-tg-sent4', 'r')

gr7_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr7-h0.1-thtg-loss', 'r')
gr7_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr7-h0.1-thtg-sent1', 'r')
gr7_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr7-h0.1-thtg-sent2', 'r')
gr7_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr7-h0.1-thtg-sent3', 'r')
gr7_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr7-h0.1-thtg-sent4', 'r')
gr7_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr7-h0-tg-loss', 'r')
gr7_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr7-h0-tg-sent1', 'r')
gr7_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr7-h0-tg-sent2', 'r')
gr7_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr7-h0-tg-sent3', 'r')
gr7_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr7-h0-tg-sent4', 'r')

gr8_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr8-h0.1-thtg-loss', 'r')
gr8_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr8-h0.1-thtg-sent1', 'r')
gr8_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr8-h0.1-thtg-sent2', 'r')
gr8_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr8-h0.1-thtg-sent3', 'r')
gr8_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr8-h0.1-thtg-sent4', 'r')
gr8_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr8-h0-tg-loss', 'r')
gr8_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr8-h0-tg-sent1', 'r')
gr8_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr8-h0-tg-sent2', 'r')
gr8_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr8-h0-tg-sent3', 'r')
gr8_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr8-h0-tg-sent4', 'r')

gr9_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr9-h0.1-thtg-loss', 'r')
gr9_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr9-h0.1-thtg-sent1', 'r')
gr9_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr9-h0.1-thtg-sent2', 'r')
gr9_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr9-h0.1-thtg-sent3', 'r')
gr9_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr9-h0.1-thtg-sent4', 'r')
gr9_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr9-h0-tg-loss', 'r')
gr9_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr9-h0-tg-sent1', 'r')
gr9_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr9-h0-tg-sent2', 'r')
gr9_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr9-h0-tg-sent3', 'r')
gr9_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr9-h0-tg-sent4', 'r')

gr10_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr10-h0.1-thtg-loss', 'r')
gr10_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr10-h0.1-thtg-sent1', 'r')
gr10_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr10-h0.1-thtg-sent2', 'r')
gr10_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr10-h0.1-thtg-sent3', 'r')
gr10_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr10-h0.1-thtg-sent4', 'r')
gr10_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr10-h0-tg-loss', 'r')
gr10_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr10-h0-tg-sent1', 'r')
gr10_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr10-h0-tg-sent2', 'r')
gr10_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr10-h0-tg-sent3', 'r')
gr10_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr10-h0-tg-sent4', 'r')

gr11_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr11-h0.1-thtg-loss', 'r')
gr11_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr11-h0.1-thtg-sent1', 'r')
gr11_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr11-h0.1-thtg-sent2', 'r')
gr11_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr11-h0.1-thtg-sent3', 'r')
gr11_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr11-h0.1-thtg-sent4', 'r')
gr11_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr11-h0-tg-loss', 'r')
gr11_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr11-h0-tg-sent1', 'r')
gr11_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr11-h0-tg-sent2', 'r')
gr11_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr11-h0-tg-sent3', 'r')
gr11_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr11-h0-tg-sent4', 'r')

gr12_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr12-h0.1-thtg-loss', 'r')
gr12_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr12-h0.1-thtg-sent1', 'r')
gr12_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr12-h0.1-thtg-sent2', 'r')
gr12_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr12-h0.1-thtg-sent3', 'r')
gr12_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr12-h0.1-thtg-sent4', 'r')
gr12_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr12-h0-tg-loss', 'r')
gr12_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr12-h0-tg-sent1', 'r')
gr12_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr12-h0-tg-sent2', 'r')
gr12_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr12-h0-tg-sent3', 'r')
gr12_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr12-h0-tg-sent4', 'r')

gr13_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr13-h0.1-thtg-loss', 'r')
gr13_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr13-h0.1-thtg-sent1', 'r')
gr13_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr13-h0.1-thtg-sent2', 'r')
gr13_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr13-h0.1-thtg-sent3', 'r')
gr13_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr13-h0.1-thtg-sent4', 'r')
gr13_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr13-h0-tg-loss', 'r')
gr13_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr13-h0-tg-sent1', 'r')
gr13_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr13-h0-tg-sent2', 'r')
gr13_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr13-h0-tg-sent3', 'r')
gr13_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr13-h0-tg-sent4', 'r')

gr14_thtg_loss = open('D:\PyCharm5.0.3\project\data19\gr14-h0.1-thtg-loss', 'r')
gr14_thtg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr14-h0.1-thtg-sent1', 'r')
gr14_thtg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr14-h0.1-thtg-sent2', 'r')
gr14_thtg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr14-h0.1-thtg-sent3', 'r')
gr14_thtg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr14-h0.1-thtg-sent4', 'r')
gr14_tg_loss = open('D:\PyCharm5.0.3\project\data19\gr14-h0-tg-loss', 'r')
gr14_tg_sent1 = open('D:\PyCharm5.0.3\project\data19\gr14-h0-tg-sent1', 'r')
gr14_tg_sent2 = open('D:\PyCharm5.0.3\project\data19\gr14-h0-tg-sent2', 'r')
gr14_tg_sent3 = open('D:\PyCharm5.0.3\project\data19\gr14-h0-tg-sent3', 'r')
gr14_tg_sent4 = open('D:\PyCharm5.0.3\project\data19\gr14-h0-tg-sent4', 'r')


def read_line(data):
    for line in data:
        data_list = eval(line)
    return data_list

thtg_loss_sample = [read_line(gr0_thtg_loss), read_line(gr1_thtg_loss), read_line(gr2_thtg_loss), read_line(gr3_thtg_loss),
                    read_line(gr4_thtg_loss), read_line(gr5_thtg_loss), read_line(gr6_thtg_loss), read_line(gr7_thtg_loss),
                    read_line(gr9_thtg_loss), read_line(gr10_thtg_loss), read_line(gr11_thtg_loss),
                    read_line(gr12_thtg_loss), read_line(gr13_thtg_loss), read_line(gr14_thtg_loss)]
thtg_sent1_sample = [read_line(gr0_thtg_sent1), read_line(gr1_thtg_sent1), read_line(gr2_thtg_sent1), read_line(gr3_thtg_sent1),
                     read_line(gr4_thtg_sent1), read_line(gr5_thtg_sent1), read_line(gr6_thtg_sent1), read_line(gr7_thtg_sent1),
                     read_line(gr9_thtg_sent1), read_line(gr10_thtg_sent1), read_line(gr11_thtg_sent1),
                     read_line(gr12_thtg_sent1), read_line(gr13_thtg_sent1), read_line(gr14_thtg_sent1)]
thtg_sent2_sample = [read_line(gr0_thtg_sent2), read_line(gr1_thtg_sent2), read_line(gr2_thtg_sent2), read_line(gr3_thtg_sent2),
                     read_line(gr4_thtg_sent2), read_line(gr5_thtg_sent2), read_line(gr6_thtg_sent2), read_line(gr7_thtg_sent2),
                     read_line(gr9_thtg_sent2), read_line(gr10_thtg_sent2), read_line(gr11_thtg_sent2),
                     read_line(gr12_thtg_sent2), read_line(gr13_thtg_sent2), read_line(gr14_thtg_sent2)]
thtg_sent3_sample = [read_line(gr0_thtg_sent3), read_line(gr1_thtg_sent3), read_line(gr2_thtg_sent3), read_line(gr3_thtg_sent3),
                     read_line(gr4_thtg_sent3), read_line(gr5_thtg_sent3), read_line(gr6_thtg_sent3), read_line(gr7_thtg_sent3),
                     read_line(gr9_thtg_sent3), read_line(gr10_thtg_sent3), read_line(gr11_thtg_sent3),
                     read_line(gr12_thtg_sent3), read_line(gr13_thtg_sent3), read_line(gr14_thtg_sent3)]
thtg_sent4_sample = [read_line(gr0_thtg_sent4), read_line(gr1_thtg_sent4), read_line(gr2_thtg_sent4), read_line(gr3_thtg_sent4),
                     read_line(gr4_thtg_sent4), read_line(gr5_thtg_sent4), read_line(gr6_thtg_sent4), read_line(gr7_thtg_sent4),
                     read_line(gr9_thtg_sent4), read_line(gr10_thtg_sent4), read_line(gr11_thtg_sent4),
                     read_line(gr12_thtg_sent4), read_line(gr13_thtg_sent4), read_line(gr14_thtg_sent4)]

tg_loss_sample = [read_line(gr0_tg_loss), read_line(gr1_tg_loss), read_line(gr2_tg_loss), read_line(gr3_tg_loss),
                  read_line(gr4_tg_loss), read_line(gr5_tg_loss), read_line(gr6_tg_loss), read_line(gr7_tg_loss),
                  read_line(gr9_tg_loss), read_line(gr10_tg_loss), read_line(gr11_tg_loss),
                  read_line(gr12_tg_loss), read_line(gr13_tg_loss), read_line(gr14_tg_loss)]
tg_sent1_sample = [read_line(gr0_tg_sent1), read_line(gr1_tg_sent1), read_line(gr2_tg_sent1), read_line(gr3_tg_sent1),
                   read_line(gr4_tg_sent1), read_line(gr5_tg_sent1), read_line(gr6_tg_sent1), read_line(gr7_tg_sent1),
                   read_line(gr9_tg_sent1), read_line(gr10_tg_sent1), read_line(gr11_tg_sent1),
                   read_line(gr12_tg_sent1), read_line(gr13_tg_sent1), read_line(gr14_tg_sent1)]
tg_sent2_sample = [read_line(gr0_tg_sent2), read_line(gr1_tg_sent2), read_line(gr2_tg_sent2), read_line(gr3_tg_sent2),
                   read_line(gr4_tg_sent2), read_line(gr5_tg_sent2), read_line(gr6_tg_sent2), read_line(gr7_tg_sent2),
                   read_line(gr9_tg_sent2), read_line(gr10_tg_sent2), read_line(gr11_tg_sent2),
                   read_line(gr12_tg_sent2), read_line(gr13_tg_sent2), read_line(gr14_tg_sent2)]
tg_sent3_sample = [read_line(gr0_tg_sent3), read_line(gr1_tg_sent3), read_line(gr2_tg_sent3), read_line(gr3_tg_sent3),
                   read_line(gr4_tg_sent3), read_line(gr5_tg_sent3), read_line(gr6_tg_sent3), read_line(gr7_tg_sent3),
                   read_line(gr9_tg_sent3), read_line(gr10_tg_sent3), read_line(gr11_tg_sent3),
                   read_line(gr12_tg_sent3), read_line(gr13_tg_sent3), read_line(gr14_tg_sent3)]
tg_sent4_sample = [read_line(gr0_tg_sent4), read_line(gr1_tg_sent4), read_line(gr2_tg_sent4), read_line(gr3_tg_sent4),
                   read_line(gr4_tg_sent4), read_line(gr5_tg_sent4), read_line(gr6_tg_sent4), read_line(gr7_tg_sent4),
                   read_line(gr9_tg_sent4), read_line(gr10_tg_sent4), read_line(gr11_tg_sent4),
                   read_line(gr12_tg_sent4), read_line(gr13_tg_sent4), read_line(gr14_tg_sent4)]

update_list = np.linspace(0, 200, 201)

classical_loss_sample = read_line(classical_loss)
classical_sent1_sample = read_line(classical_sent1)
classical_sent2_sample = read_line(classical_sent2)
classical_sent3_sample = read_line(classical_sent3)
classical_sent4_sample = read_line(classical_sent4)

classical_loss_sample_for_update = [[] for i in range(0, len(update_list))]
classical_loss_sample_ave = [0 for i in range(0, len(update_list))]
classical_loss_sample_var = [0 for i in range(0, len(update_list))]

classical_sent1_sample_for_update = [[] for i in range(0, len(update_list))]
classical_sent1_sample_ave = [0 for i in range(0, len(update_list))]
classical_sent1_sample_var = [0 for i in range(0, len(update_list))]

classical_sent2_sample_for_update = [[] for i in range(0, len(update_list))]
classical_sent2_sample_ave = [0 for i in range(0, len(update_list))]
classical_sent2_sample_var = [0 for i in range(0, len(update_list))]

classical_sent3_sample_for_update = [[] for i in range(0, len(update_list))]
classical_sent3_sample_ave = [0 for i in range(0, len(update_list))]
classical_sent3_sample_var = [0 for i in range(0, len(update_list))]

classical_sent4_sample_for_update = [[] for i in range(0, len(update_list))]
classical_sent4_sample_ave = [0 for i in range(0, len(update_list))]
classical_sent4_sample_var = [0 for i in range(0, len(update_list))]

classical_sentave_sample_ave = [0 for i in range(0, len(update_list))]
classical_sentave_sample_var = [0 for i in range(0, len(update_list))]

thtg_loss_sample_for_update = [[] for i in range(0, len(update_list))]
thtg_loss_sample_ave = [0 for i in range(0, len(update_list))]
thtg_loss_sample_var = [0 for i in range(0, len(update_list))]


thtg_sent1_sample_for_update = [[] for i in range(0, len(update_list))]
thtg_sent1_sample_ave = [0 for i in range(0, len(update_list))]
thtg_sent1_sample_var = [0 for i in range(0, len(update_list))]

thtg_sent2_sample_for_update = [[] for i in range(0, len(update_list))]
thtg_sent2_sample_ave = [0 for i in range(0, len(update_list))]
thtg_sent2_sample_var = [0 for i in range(0, len(update_list))]

thtg_sent3_sample_for_update = [[] for i in range(0, len(update_list))]
thtg_sent3_sample_ave = [0 for i in range(0, len(update_list))]
thtg_sent3_sample_var = [0 for i in range(0, len(update_list))]

thtg_sent4_sample_for_update = [[] for i in range(0, len(update_list))]
thtg_sent4_sample_ave = [0 for i in range(0, len(update_list))]
thtg_sent4_sample_var = [0 for i in range(0, len(update_list))]


thtg_sentave_sample_ave = [0 for i in range(0, len(update_list))]
thtg_sentave_sample_var = [0 for i in range(0, len(update_list))]

th_loss_sample_for_update = [[] for i in range(0, len(update_list))]
th_loss_sample_ave = [0 for i in range(0, len(update_list))]
th_loss_sample_var = [0 for i in range(0, len(update_list))]

th_sent1_sample_for_update = [[] for i in range(0, len(update_list))]
th_sent1_sample_ave = [0 for i in range(0, len(update_list))]
th_sent1_sample_var = [0 for i in range(0, len(update_list))]

th_sent2_sample_for_update = [[] for i in range(0, len(update_list))]
th_sent2_sample_ave = [0 for i in range(0, len(update_list))]
th_sent2_sample_var = [0 for i in range(0, len(update_list))]

tg_loss_sample_for_update = [[] for i in range(0, len(update_list))]
tg_loss_sample_ave = [0 for i in range(0, len(update_list))]
tg_loss_sample_var = [0 for i in range(0, len(update_list))]

tg_sent1_sample_for_update = [[] for i in range(0, len(update_list))]
tg_sent1_sample_ave = [0 for i in range(0, len(update_list))]
tg_sent1_sample_var = [0 for i in range(0, len(update_list))]

tg_sent2_sample_for_update = [[] for i in range(0, len(update_list))]
tg_sent2_sample_ave = [0 for i in range(0, len(update_list))]
tg_sent2_sample_var = [0 for i in range(0, len(update_list))]

tg_sent3_sample_for_update = [[] for i in range(0, len(update_list))]
tg_sent3_sample_ave = [0 for i in range(0, len(update_list))]
tg_sent3_sample_var = [0 for i in range(0, len(update_list))]

tg_sent4_sample_for_update = [[] for i in range(0, len(update_list))]
tg_sent4_sample_ave = [0 for i in range(0, len(update_list))]
tg_sent4_sample_var = [0 for i in range(0, len(update_list))]

tg_sentave_sample_ave = [0 for i in range(0, len(update_list))]
tg_sentave_sample_var = [0 for i in range(0, len(update_list))]

for update_num in range(0, len(update_list)):
    for sample in classical_loss_sample:
        classical_loss_sample_for_update[update_num].append(sample[update_num])
    classical_loss_sample_ave[update_num] = np.mean(classical_loss_sample_for_update[update_num])
    classical_loss_sample_var[update_num] = np.var(classical_loss_sample_for_update[update_num])

    for sample in classical_sent1_sample:
        classical_sent1_sample_for_update[update_num].append(sample[update_num])
    classical_sent1_sample_ave[update_num] = np.mean(classical_sent1_sample_for_update[update_num])
    classical_sent1_sample_var[update_num] = np.var(classical_sent1_sample_for_update[update_num])

    for sample in classical_sent2_sample:
        classical_sent2_sample_for_update[update_num].append(sample[update_num])
    classical_sent2_sample_ave[update_num] = np.mean(classical_sent2_sample_for_update[update_num])
    classical_sent2_sample_var[update_num] = np.var(classical_sent2_sample_for_update[update_num])

    for sample in classical_sent3_sample:
        classical_sent3_sample_for_update[update_num].append(sample[update_num])
    classical_sent3_sample_ave[update_num] = np.mean(classical_sent3_sample_for_update[update_num])
    classical_sent3_sample_var[update_num] = np.var(classical_sent3_sample_for_update[update_num])

    for sample in classical_sent4_sample:
        classical_sent4_sample_for_update[update_num].append(sample[update_num])
    classical_sent4_sample_ave[update_num] = np.mean(classical_sent4_sample_for_update[update_num])
    classical_sent4_sample_var[update_num] = np.var(classical_sent4_sample_for_update[update_num])

    classical_sentave_sample_ave[update_num] = np.mean([classical_sent1_sample_ave[update_num], classical_sent2_sample_ave[update_num], classical_sent3_sample_ave[update_num], classical_sent4_sample_ave[update_num]])
    classical_sentave_sample_var[update_num] = np.mean([classical_sent1_sample_var[update_num], classical_sent2_sample_var[update_num], classical_sent3_sample_var[update_num], classical_sent4_sample_var[update_num]])

    for sample in thtg_loss_sample:
        thtg_loss_sample_for_update[update_num].append(sample[update_num])
    thtg_loss_sample_ave[update_num] = np.mean(thtg_loss_sample_for_update[update_num])
    thtg_loss_sample_var[update_num] = np.var(thtg_loss_sample_for_update[update_num])

    for sample in thtg_sent1_sample:
        thtg_sent1_sample_for_update[update_num].append(sample[update_num])
    thtg_sent1_sample_ave[update_num] = np.mean(thtg_sent1_sample_for_update[update_num])
    thtg_sent1_sample_var[update_num] = np.var(thtg_sent1_sample_for_update[update_num])

    for sample in thtg_sent2_sample:
        thtg_sent2_sample_for_update[update_num].append(sample[update_num])
    thtg_sent2_sample_ave[update_num] = np.mean(thtg_sent2_sample_for_update[update_num])
    thtg_sent2_sample_var[update_num] = np.var(thtg_sent2_sample_for_update[update_num])

    for sample in thtg_sent3_sample:
        thtg_sent3_sample_for_update[update_num].append(sample[update_num])
    thtg_sent3_sample_ave[update_num] = np.mean(thtg_sent3_sample_for_update[update_num])
    thtg_sent3_sample_var[update_num] = np.var(thtg_sent3_sample_for_update[update_num])

    for sample in thtg_sent4_sample:
        thtg_sent4_sample_for_update[update_num].append(sample[update_num])
    thtg_sent4_sample_ave[update_num] = np.mean(thtg_sent4_sample_for_update[update_num])
    thtg_sent4_sample_var[update_num] = np.var(thtg_sent4_sample_for_update[update_num])

    thtg_sentave_sample_ave[update_num] = np.mean([thtg_sent1_sample_ave[update_num], thtg_sent2_sample_ave[update_num], thtg_sent3_sample_ave[update_num], thtg_sent4_sample_ave[update_num]])
    thtg_sentave_sample_var[update_num] = np.mean([thtg_sent1_sample_var[update_num], thtg_sent2_sample_var[update_num], thtg_sent3_sample_var[update_num], thtg_sent4_sample_var[update_num]])

    # for sample in th_loss_sample:
    #     th_loss_sample_for_update[update_num].append(sample[update_num])
    # th_loss_sample_ave[update_num] = np.mean(th_loss_sample_for_update[update_num])
    # th_loss_sample_var[update_num] = np.var(th_loss_sample_for_update[update_num])
    #
    # for sample in th_sent1_sample:
    #     th_sent1_sample_for_update[update_num].append(sample[update_num])
    # th_sent1_sample_ave[update_num] = np.mean(th_sent1_sample_for_update[update_num])
    # th_sent1_sample_var[update_num] = np.var(th_sent1_sample_for_update[update_num])
    #
    # for sample in th_sent2_sample:
    #     th_sent2_sample_for_update[update_num].append(sample[update_num])
    # th_sent2_sample_ave[update_num] = np.mean(th_sent2_sample_for_update[update_num])
    # th_sent2_sample_var[update_num] = np.var(th_sent2_sample_for_update[update_num])

    for sample in tg_loss_sample:
        tg_loss_sample_for_update[update_num].append(sample[update_num])
    tg_loss_sample_ave[update_num] = np.mean(tg_loss_sample_for_update[update_num])
    tg_loss_sample_var[update_num] = np.var(tg_loss_sample_for_update[update_num])

    for sample in tg_sent1_sample:
        tg_sent1_sample_for_update[update_num].append(sample[update_num])
    tg_sent1_sample_ave[update_num] = np.mean(tg_sent1_sample_for_update[update_num])
    tg_sent1_sample_var[update_num] = np.var(tg_sent1_sample_for_update[update_num])

    for sample in tg_sent2_sample:
        tg_sent2_sample_for_update[update_num].append(sample[update_num])
    tg_sent2_sample_ave[update_num] = np.mean(tg_sent2_sample_for_update[update_num])
    tg_sent2_sample_var[update_num] = np.var(tg_sent2_sample_for_update[update_num])

    for sample in tg_sent3_sample:
        tg_sent3_sample_for_update[update_num].append(sample[update_num])
    tg_sent3_sample_ave[update_num] = np.mean(tg_sent3_sample_for_update[update_num])
    tg_sent3_sample_var[update_num] = np.var(tg_sent3_sample_for_update[update_num])

    for sample in tg_sent4_sample:
        tg_sent4_sample_for_update[update_num].append(sample[update_num])
    tg_sent4_sample_ave[update_num] = np.mean(tg_sent4_sample_for_update[update_num])
    tg_sent4_sample_var[update_num] = np.var(tg_sent4_sample_for_update[update_num])

    tg_sentave_sample_ave[update_num] = np.mean([tg_sent1_sample_ave[update_num], tg_sent2_sample_ave[update_num], tg_sent3_sample_ave[update_num], tg_sent4_sample_ave[update_num]])
    tg_sentave_sample_var[update_num] = np.mean([tg_sent1_sample_var[update_num], tg_sent2_sample_var[update_num], tg_sent3_sample_var[update_num], tg_sent4_sample_var[update_num]])

font_leg = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 15,
         }
# print(tg_sent2_sample_var)
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 23,
         }
'''
# 测试每个样本
update_list = np.linspace(0, 200, 201)
fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(update_list, read_line(gr0_thtg_loss), label='thtg', color='blue')
ax1.plot(update_list, read_line(gr0_tg_loss), label='tg', color='red')
plt.legend()
ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(update_list, read_line(gr0_thtg_sent1), label='thtg', color='blue')
ax2.plot(update_list, read_line(gr0_tg_sent1), label='tg', color='red')
plt.legend()
ax3 = fig.add_subplot(1, 3, 3)
ax3.plot(update_list, read_line(gr0_thtg_sent2), label='thtg', color='blue')
ax3.plot(update_list, read_line(gr0_tg_sent2), label='tg', color='red')
plt.legend()
plt.show()

'''
# 画图参数
cap_size = 2
eline_width = 1.5
line_width = 3.5

fig = plt.figure(figsize=(9, 20))
ax1 = fig.add_subplot(3, 2, 1)
ax1.errorbar(update_list, thtg_loss_sample_ave, yerr=thtg_loss_sample_var, fmt="-", linewidth=line_width,
             ecolor='royalblue', color='blue', capsize=cap_size, label='coherent', elinewidth=eline_width, errorevery=10)
# ax1.scatter(get_element(update_list, 20), get_element(thtg_loss_sample_ave, 20), marker='^', s=28, color='blue')
ax1.errorbar(update_list, tg_loss_sample_ave, yerr=tg_loss_sample_var, fmt="--", linewidth=line_width,
             ecolor='tomato', color='red', capsize=cap_size, label='incoherent', elinewidth=eline_width, errorevery=10)
# ax1.scatter(get_element(update_list, 20), get_element(tg_loss_sample_ave, 20), marker='s', s=28, color='red')
ax1.errorbar(update_list, classical_loss_sample_ave, yerr=classical_loss_sample_var, fmt=":", linewidth=line_width+0.5,
             ecolor='gray', color='dimgray', capsize=cap_size, label='classical', elinewidth=eline_width, errorevery=10)
# ax1.scatter(get_element(update_list, 20), get_element(classical_loss_sample_ave, 20), marker='o', s=28, color='grey')

plt.tick_params(labelsize=17)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()  # 设置刻度大小
[label.set_fontname('Times New Roman') for label in labels]   # 设置刻度字体
plt.xlabel('iterations', font2)
plt.ylabel('Loss', font2)
plt.legend(frameon=False, prop=font_leg)

ax2 = fig.add_subplot(3, 2, 2)
ax2.errorbar(update_list, thtg_sentave_sample_ave, yerr=thtg_sentave_sample_var, fmt="-", linewidth=line_width,
             ecolor='royalblue', color='blue', capsize=cap_size, label='coherent', elinewidth=eline_width, errorevery=10)
# ax2.scatter(get_element(update_list, 20), get_element(thtg_sentave_sample_ave, 20), marker='^', s=28, color='blue')
ax2.errorbar(update_list, tg_sentave_sample_ave, yerr=tg_sentave_sample_var, fmt="--", linewidth=line_width,
             ecolor='tomato', color='red', capsize=cap_size, label='incoherent', elinewidth=eline_width, errorevery=10)
# ax2.scatter(get_element(update_list, 20), get_element(tg_sentave_sample_ave, 20), marker='s', s=28, color='red')
ax2.errorbar(update_list, classical_sentave_sample_ave, yerr=classical_sentave_sample_var, fmt=":", linewidth=line_width+0.5,
             ecolor='gray', color='dimgray', capsize=cap_size, label='classical', elinewidth=eline_width, errorevery=10)
# ax2.scatter(get_element(update_list, 20), get_element(classical_sentave_sample_ave, 20), marker='o', s=28, color='grey')
ax2.set_ylim(0.2, 1.05)
plt.tick_params(labelsize=17)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()  # 设置刻度大小
[label.set_fontname('Times New Roman') for label in labels]   # 设置刻度字体
plt.xlabel('iterations', font2)
plt.ylabel('average accuracy', font2)
plt.legend(frameon=False, prop=font_leg)


ax3 = fig.add_subplot(3, 2, 3)
ax3.errorbar(update_list, thtg_sent1_sample_ave, yerr=thtg_sent1_sample_var, fmt="-", linewidth=line_width,
             ecolor='royalblue', color='blue', capsize=cap_size, label='coherent', elinewidth=eline_width, errorevery=10)
# ax3.scatter(get_element(update_list, 20), get_element(thtg_sent1_sample_ave, 20), marker='^', s=28, color='blue')
ax3.errorbar(update_list, tg_sent1_sample_ave, yerr=tg_sent1_sample_var, fmt="--", linewidth=line_width,
             ecolor='tomato', color='red', capsize=cap_size, label='incoherent', elinewidth=eline_width, errorevery=10)
# ax3.scatter(get_element(update_list, 20), get_element(tg_sent1_sample_ave, 20), marker='s', s=28, color='red')
ax3.errorbar(update_list, classical_sent1_sample_ave, yerr=classical_sent1_sample_var, fmt=':', linewidth=line_width+0.5,
             ecolor='gray', color='dimgray', capsize=cap_size, label='classical', elinewidth=eline_width, errorevery=10)
# ax3.scatter(get_element(update_list, 20), get_element(classical_sent1_sample_ave, 20), marker='o', s=28, color='grey')

plt.tick_params(labelsize=17)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()  # 设置刻度大小
[label.set_fontname('Times New Roman') for label in labels]   # 设置刻度字体
plt.xlabel('iterations', font2)
plt.ylabel('verse1 accuracy', font2)
plt.legend(frameon=False, prop=font_leg)

ax4 = fig.add_subplot(3, 2, 4)
ax4.errorbar(update_list, thtg_sent2_sample_ave, yerr=thtg_sent2_sample_var, fmt="-", linewidth=line_width,
             ecolor='royalblue', color='blue', capsize=cap_size, label='coherent', elinewidth=eline_width, errorevery=10)
# ax4.scatter(get_element(update_list, 20), get_element(thtg_sent2_sample_ave, 20), marker='^', s=28, color='blue')
ax4.errorbar(update_list, tg_sent2_sample_ave, yerr=tg_sent2_sample_var, fmt="--", linewidth=line_width,
             ecolor='tomato', color='red', capsize=cap_size, label='incoherent', elinewidth=eline_width, errorevery=10)
# ax4.scatter(get_element(update_list, 20), get_element(tg_sent2_sample_ave, 20), marker='s', s=28, color='red')
ax4.errorbar(update_list, classical_sent2_sample_ave, yerr=classical_sent2_sample_var, fmt=":", linewidth=line_width+0.5,
             ecolor='gray', color='dimgray', capsize=cap_size, label='classical', elinewidth=eline_width, errorevery=10)
# ax4.scatter(get_element(update_list, 20), get_element(classical_sent2_sample_ave, 20), marker='o', s=28, color='grey')
plt.tick_params(labelsize=17)
labels = ax4.get_xticklabels() + ax4.get_yticklabels()  # 设置刻度大小
[label.set_fontname('Times New Roman') for label in labels]   # 设置刻度字体
plt.xlabel('iterations', font2)
plt.ylabel('verse2 accuracy', font2)
plt.legend(frameon=False, prop=font_leg)

ax5 = fig.add_subplot(3, 2, 5)
ax5.errorbar(update_list, thtg_sent3_sample_ave, yerr=thtg_sent3_sample_var, fmt="-", linewidth=line_width,
             ecolor='royalblue', color='blue', capsize=cap_size, label='coherent', elinewidth=eline_width, errorevery=10)
# ax5.scatter(get_element(update_list, 20), get_element(thtg_sent3_sample_ave, 20), marker='^', s=28, color='blue')
ax5.errorbar(update_list, tg_sent3_sample_ave, yerr=tg_sent3_sample_var, fmt="--", linewidth=line_width,
             ecolor='tomato', color='red', capsize=cap_size, label='incoherent', elinewidth=eline_width, errorevery=10)
# ax5.scatter(get_element(update_list, 20), get_element(tg_sent3_sample_ave, 20), marker='s', s=28, color='red')
ax5.errorbar(update_list, classical_sent3_sample_ave, yerr=classical_sent3_sample_var, fmt=":", linewidth=line_width+0.5,
             ecolor='gray', color='dimgray', capsize=cap_size, label='classical', elinewidth=eline_width, errorevery=10)
# ax5.scatter(get_element(update_list, 20), get_element(classical_sent3_sample_ave, 20), marker='o', s=28, color='grey')
ax5.set_ylim(0.2, 1.05)
plt.tick_params(labelsize=17)
labels = ax5.get_xticklabels() + ax5.get_yticklabels()  # 设置刻度大小
[label.set_fontname('Times New Roman') for label in labels]   # 设置刻度字体
plt.xlabel('iterations', font2)
plt.ylabel('sentence1 accuracy', font2)
plt.legend(frameon=False, prop=font_leg)

ax6 = fig.add_subplot(3, 2, 6)
ax6.errorbar(update_list, thtg_sent4_sample_ave, yerr=thtg_sent4_sample_var, fmt="-", linewidth=line_width,
             ecolor='royalblue', color='blue', capsize=cap_size, label='coherent', elinewidth=eline_width, errorevery=10)
# ax6.scatter(get_element(update_list, 20), get_element(thtg_sent4_sample_ave, 20), marker='^', s=28, color='blue')
ax6.errorbar(update_list, tg_sent4_sample_ave, yerr=tg_sent4_sample_var, fmt="--", linewidth=line_width,
             ecolor='tomato', color='red', capsize=cap_size, label='incoherent', elinewidth=eline_width, errorevery=10)
# ax6.scatter(get_element(update_list, 20), get_element(tg_sent4_sample_ave, 20), marker='s', s=28, color='red')
ax6.errorbar(update_list, classical_sent4_sample_ave, yerr=classical_sent4_sample_var, fmt=":", linewidth=line_width+0.5,
             ecolor='gray', color='dimgray', capsize=cap_size, label='classical', elinewidth=eline_width, errorevery=10)
# ax6.scatter(get_element(update_list, 20), get_element(classical_sent4_sample_ave, 20), marker='o', s=28, color='grey')
ax6.set_ylim(0.2, 1.05)
plt.tick_params(labelsize=17)
labels = ax6.get_xticklabels() + ax6.get_yticklabels()  # 设置刻度大小
[label.set_fontname('Times New Roman') for label in labels]   # 设置刻度字体
plt.xlabel('iterations', font2)
plt.ylabel('sentence2 accuracy', font2)
plt.legend(frameon=False, prop=font_leg)



plt.show()