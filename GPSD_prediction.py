# -*- coding: utf-8 -*-
"""
@author: Cheng Han
"""


import os
import math
import numpy as np
from collections import Counter
from keras.models import load_model
import joblib
import gc


# GPS
def readPeptide(pepfile, lr):
    data = []
    lr = 30 - lr
    with open(pepfile, 'r') as f:
        for line in f:
            if lr == 0:
                data.append(line.rstrip().split('\t')[0])
            else:
                data.append(line.rstrip().split('\t')[0][lr:-lr])
    return data


def readweight(weight_file):
    weight = None
    with open(weight_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 2 - 1:
                weight = np.array([float(x) for x in line.rstrip().split('\t')])
    return weight


class GpsPredictor(object):
    def __init__(self, plist, pls_weight, mm_weight):
        '''
        initial GPS predictor using positive training set, pls_weight vector and mm_weight vector
        :param plist: (list) positive peptides list
        :param pls_weight:  (list) pls_weight vector
        :param mm_weight:   (list) mm_weight vector
        '''
        self.alist = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
                      'P', 'S', 'T', 'W', 'Y', 'V', '*']
        self.plist = plist
        self.pls_weight = np.array(pls_weight).flatten()
        self.mm_weight = np.array(mm_weight).flatten()

        self.__count_matrix = self._plist_index()
        self.__mm_matrix, self.__mm_intercept = self._mmweight2matrix()

    def predict(self, query_peptide, loo=False):
        '''
        return the gps score for the query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: gps score
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(self.__count_matrix)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        pls_count_matrix = (count_clone.T * self.pls_weight).T / (len(self.plist) - rm_num)
        return np.sum(matrix * pls_count_matrix) + self.__mm_intercept

    def generatePLSdata(self, query_peptide, loo=False):
        '''
        generate the pls vector of query peptide
        :param query_peptide: (str) query peptide
        :param loo: (bool) if true, the count_matrix will minus 1 according to the amino acid in each position in query peptide
        :return: (np.ndarray) the vector of feature for each position
        '''
        count_clone = self.__count_matrix * len(self.plist)
        matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo:
                count_clone[i, self.alist.index(a)] -= 1

            matrix[i, :] = self.__mm_matrix[self.alist.index(a), :]
        rm_num = 1 if loo else 0
        count_clone = (count_clone.T * self.pls_weight).T
        return np.sum(matrix * count_clone / (len(self.plist) - rm_num), 1)

    def generateMMdata(self, query_peptide, loo=False):
        count_clone = self.__count_matrix * len(self.plist)

        indicator_matrix = np.zeros_like(count_clone)
        for i, a in enumerate(query_peptide):
            if a not in self.alist: a = 'C'
            if loo: count_clone[i, self.alist.index(a)] -= 1
            indicator_matrix[i, self.alist.index(a)] = 1

        rm_num = 1 if loo else 0

        count_clone /= (len(self.plist) - rm_num)

        pls_count_matrix = (count_clone.T * self.pls_weight).T

        m = np.dot(indicator_matrix.T, pls_count_matrix) * self.__mm_matrix

        m += m.T

        np.fill_diagonal(m, np.diag(m) / float(2))

        iu1 = np.triu_indices(m.shape[0])

        return m[iu1]

    def getcutoff(self, randompeplist, sp=[0.98, 0.95, 0.85]):
        '''
        return cutoffs using 10000 random peptides as negative
        :param randompeplist: (list) random generated peptides
        :param sp: (float list) sp to be used for cutoff setting
        :return: (float list) cutoffs, same lens with sp
        '''
        rand_scores = sorted([self.predict(p) for p in randompeplist])
        cutoffs = np.zeros(len(sp))
        for i, s in enumerate(sp):
            index = np.floor(len(rand_scores) * s).astype(int)
            cutoffs[i] = rand_scores[index]
        return cutoffs

    def _plist_index(self):
        '''
        return the amino acid frequency on each position, row: position, column: self.alist, 61 x 24
        :return: count matrix
        '''
        n, m = len(self.plist[0]), len(self.alist)
        count_matrix = np.zeros((n, m))
        for i in range(n):
            for p in self.plist:
                count_matrix[i][self.alist.index(p[i])] += 1
        return count_matrix / float(len(self.plist))

    def _mmweight2matrix(self):
        '''
        convert matrix weight vector to similarity matrix, 24 x 24, index order is self.alist
        :return:
        '''
        aalist = self.getaalist()
        mm_matrix = np.zeros((len(self.alist), len(self.alist)))
        for n, d in enumerate(aalist):
            value = self.mm_weight[n + 1]  # mm weight contain intercept
            i, j = self.alist.index(d[0]), self.alist.index(d[1])
            mm_matrix[i, j] = value
            mm_matrix[j, i] = value
        return mm_matrix, self.mm_weight[0]

    def getaalist(self):
        '''return aa-aa list
        AA: 0
        AR: 1
        '''
        aa = [self.alist[i] + self.alist[j] for i in range(len(self.alist)) for j in range(i, len(self.alist))]
        return aa


# qlist query list for all peptide
def gps(qlist, PN):
    # global gpn

    def generateMMData(querylist, plist, pls_weight, mm_weight, loo=True, positive=False):
        gp = GpsPredictor(plist, pls_weight, mm_weight)

        d = []

        for query_peptide in querylist:
            d.append(gp.generateMMdata(query_peptide, loo).tolist())
        return d

    mm_weight = readweight('models/pref/BLOSUM62R.txt')  # 1th is intercept

    ll = len(qlist[0])

    if PN == 'Y':
        Positive_pep = 'models/pref/Y_positive_list.txt'
    elif PN == 'ST':
        Positive_pep = 'models/pref/ST_positive_list.txt'

    plist = readPeptide(Positive_pep, int(ll / 2))

    gpn = generateMMData(qlist, plist, np.repeat(1, ll), mm_weight, loo=False, positive=False)

    return gpn


def get_GPS(seq, PN):
    results = gps(seq, PN)
    np_results = np.array(results)

    return np_results


amino_acid = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F',
              'P', 'S', 'T', 'W', 'Y', 'V']

aaPairs = []
for aa1 in amino_acid:
    for aa2 in amino_acid:
        aaPairs.append(aa1 + aa2)

aaDict = {}
for aa_idx in range(len(amino_acid)):
    aaDict[amino_acid[aa_idx]] = aa_idx

AAindex = [[4.35, 4.38, 4.75, 4.76, 4.65,
            4.37, 4.29, 3.97, 4.63, 3.95,
            4.17, 4.36, 4.52, 4.66, 4.44,
            4.5, 4.35, 4.7, 4.6, 3.95],
           [0.61, 0.6, 0.06, 0.46, 1.07,
            0.0, 0.47, 0.07, 0.61, 2.22,
            1.53, 1.15, 1.18, 2.02, 1.95,
            0.05, 0.05, 2.65, 1.88, 1.32],
           [1.18, 0.2, 0.23, 0.05, 1.89,
            0.72, 0.11, 0.49, 0.31, 1.45,
            3.23, 0.06, 2.67, 1.96, 0.76,
            0.97, 0.84, 0.77, 0.39, 1.08],
           [1.56, 0.45, 0.27, 0.14, 1.23,
            0.51, 0.23, 0.62, 0.29, 1.67,
            2.93, 0.15, 2.96, 2.03, 0.76,
            0.81, 0.91, 1.08, 0.68, 1.14],
           [1.0, 0.52, 0.35, 0.44, 0.06,
            0.44, 0.73, 0.35, 0.6, 0.73,
            1.0, 0.6, 1.0, 0.6, 0.06,
            0.35, 0.44, 0.73, 0.44, 0.82],
           [0.77, 0.72, 0.55, 0.65, 0.65,
            0.72, 0.55, 0.65, 0.83, 0.98,
            0.83, 0.55, 0.98, 0.98, 0.55,
            0.55, 0.83, 0.77, 0.83, 0.98],
           [0.37, 0.84, 0.97, 0.97, 0.84,
            0.64, 0.53, 0.97, 0.75, 0.37,
            0.53, 0.75, 0.64, 0.53, 0.97,
            0.84, 0.75, 0.97, 0.84, 0.37],
           [0.357, 0.529, 0.463, 0.511, 0.346,
            0.493, 0.497, 0.544, 0.323, 0.462,
            0.365, 0.466, 0.295, 0.314, 0.509,
            0.507, 0.444, 0.305, 0.42, 0.386]]

aaCodons = {'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2, 'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
            'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6, 'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
            }

Tmean = []
for pair in aaPairs:
    Tmean.append((aaCodons[pair[0]] / 61) * (aaCodons[pair[1]] / 61))

zscale = {
    'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
    'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
    'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
    'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
    'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
    'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
    'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
    'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
    'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
    'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
    'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
    'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
    'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
    'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
    'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
    'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
    'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
    'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
    'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
    'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
    '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
}


def seq2AAindex(seq):
    V_AAindex = []

    for aa in seq:
        if aa not in amino_acid:
            for j in AAindex:
                V_AAindex.append(0)
            continue
        for j in AAindex:
            V_AAindex.append(j[aaDict[aa]])

    return V_AAindex


def get_AAindex(seq):
    results = list(map(seq2AAindex, seq))
    np_results = np.array(results)

    return np_results


def seq2Binary(seq):
    V_Binary = []

    for aa in seq:
        if aa not in amino_acid:
            V_Binary = V_Binary + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            continue
        for amino in amino_acid:
            tag = 1 if aa == amino else 0
            V_Binary.append(tag)

    return V_Binary


def get_Binary(seq):
    results = list(map(seq2Binary, seq))
    np_results = np.array(results)

    return np_results


def seq2CKSAAP(seq):
    V_CKSAAP = []

    k_space = 3  # Parameter

    for k in range(k_space + 1):
        pair_sum = 0
        k_dict = {}

        for pair in aaPairs:
            k_dict[pair] = 0

        for index1 in range(len(seq)):
            index2 = index1 + k + 1
            if index1 < len(seq) and index2 < len(seq):
                if seq[index1] in amino_acid and seq[index2] in amino_acid:
                    k_dict[seq[index1] + seq[index2]] += 1
                    pair_sum = pair_sum + 1

        for pair in aaPairs:
            V_CKSAAP.append(k_dict[pair] / pair_sum)

    return V_CKSAAP


def get_CKSAAP(seq):
    results = list(map(seq2CKSAAP, seq))
    np_results = np.array(results)

    return np_results


def seq2DDE(seq):
    V_DDE = [0] * 400

    for amino in seq:
        if amino not in amino_acid:
            seq = seq.replace(amino, '')

    for i in range(len(seq) - 2 + 1):
        if seq[i] in amino_acid and seq[i + 1] in amino_acid:
            V_DDE[aaDict[seq[i]] * 20 + aaDict[seq[i + 1]]] += 1

    if sum(V_DDE) != 0:
        V_DDE = [j / sum(V_DDE) for j in V_DDE]

    Tvariance = []
    for k in range(len(Tmean)):
        Tvariance.append(Tmean[k] * (1 - Tmean[k]) / (len(seq) - 1))

    for v in range(len(V_DDE)):
        V_DDE[v] = (V_DDE[v] - Tmean[v]) / math.sqrt(Tvariance[v])

    return V_DDE


def get_DDE(seq):
    results = list(map(seq2DDE, seq))
    np_results = np.array(results)

    return np_results


def seq2DPC(seq):
    V_DPC = [0] * 400

    for amino in seq:
        if amino not in amino_acid:
            seq = seq.replace(amino, '')

    for i in range(len(seq) - 2 + 1):
        if seq[i] in amino_acid and seq[i + 1] in amino_acid:
            V_DPC[aaDict[seq[i]] * 20 + aaDict[seq[i + 1]]] += 1

    if sum(V_DPC) != 0:
        V_DPC = [j / sum(V_DPC) for j in V_DPC]

    return V_DPC


def get_DPC(seq):
    results = list(map(seq2DPC, seq))
    np_results = np.array(results)

    return np_results


def seq2EAAC(seq):
    V_EAAC = []

    sliding_window = 5  # Parameter

    for i in range(len(seq)):
        if i < len(seq) and i + sliding_window <= len(seq):
            count = Counter(seq[i:i + sliding_window])
            for key in count:
                count[key] = count[key] / len(seq[i:i + sliding_window])
            for aa in amino_acid:
                V_EAAC.append(count[aa])

    return V_EAAC


def get_EAAC(seq):
    results = list(map(seq2EAAC, seq))
    np_results = np.array(results)

    return np_results


def seq2OPF7bit1(seq):
    V_OPF7bit1 = []

    physicochemical_properties_list = [
        'ACFGHILMNPQSTVWY',
        'CFILMVW',
        'ACDGPST',
        'CFILMVWY',
        'ADGST',
        'DGNPS',
        'ACFGILVW',
    ]

    for aa in seq:
        for prop in physicochemical_properties_list:
            if aa in prop:
                V_OPF7bit1.append(1)
            else:
                V_OPF7bit1.append(0)

    return V_OPF7bit1


def get_OPF7bit1(seq):
    results = list(map(seq2OPF7bit1, seq))
    np_results = np.array(results)

    return np_results


def seq2OPF10bit(seq):
    V_OPF10bit = []

    physicochemical_properties_list = [
        'FYWH',
        'DE',
        'KHR',
        'NQSDECTKRHYW',
        'AGCTIVLKHFYWM',
        'IVL',
        'ASGC',
        'KHRDE',
        'PNDTCAGSV',
        'P',
    ]

    for aa in seq:
        for prop in physicochemical_properties_list:
            if aa in prop:
                V_OPF10bit.append(1)
            else:
                V_OPF10bit.append(0)

    return V_OPF10bit


def get_OPF10bit(seq):
    results = list(map(seq2OPF10bit, seq))
    np_results = np.array(results)

    return np_results


def seq2ZScale(seq):
    V_ZScale = []

    for aa in seq:
        if aa not in amino_acid:
            V_ZScale += zscale['-']
        else:
            V_ZScale += zscale[aa]

    return V_ZScale


def get_ZScale(seq):
    results = list(map(seq2ZScale, seq))
    np_results = np.array(results)

    return np_results


def seq2features(seq, PN):
    f1 = get_AAindex(seq)
    f2 = get_Binary(seq)
    f3 = get_CKSAAP(seq)
    f4 = get_DDE(seq)
    f5 = get_DPC(seq)
    f6 = get_EAAC(seq)
    f7 = get_GPS(seq, PN)
    f8 = get_OPF7bit1(seq)
    f9 = get_OPF10bit(seq)
    f10 = get_ZScale(seq)

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]


def test_predict(feature_ls, model_path):
    feature_list = ['AAindex', 'Binary', 'CKSAAP', 'DDE', 'DPC',
                    'EAAC', 'GPS', 'OPF7bit1', 'OPF10bit', 'ZScale']
    pred = []

    for i in range(len(feature_list)):
        f = feature_ls[i]
        dnn_model = model_path + feature_list[i] + '.h5'

        if os.path.exists(dnn_model):
            m1 = load_model(dnn_model)
            table = m1.predict(f)[:, 1]
            pred.append(table)
            del m1

        plr_model = model_path + feature_list[i] + '.pkl'
        if os.path.exists(plr_model):
            m2 = joblib.load(plr_model)
            table2 = m2.predict_proba(f)[:, 1]
            pred.append(table2)
            del m2

        gc.collect()

    pred = [[r[col] for r in pred] for col in range(len(pred[0]))]

    pred = np.array(pred)

    comb_model = model_path + 'integration.pkl'
    mc = joblib.load(comb_model)
    pred_comb = mc.predict_proba(pred)[:, 1]

    return pred_comb


def read_data(file_path):
    predict_seq = []
    with open(file_path, 'r', encoding='utf-8') as predict_file:
        for line in predict_file:
            predict_seq.append(line.strip('\r\n'))
    return predict_seq


def write_results(file_path, predict_seq, pred_comb):
    with open(file_path, 'w', encoding='utf-8') as storage_file:
        storage_file.write('Peptide\tScore\n')
        for i in range(len(predict_seq)):
            strs = predict_seq[i] + '\t' + str(pred_comb[i]) + '\n'
            storage_file.write(strs)


def main():
    
    amino_acid_type = 'ST'
    pred_file_path = 'Example_peptide.txt'
    model_path = amino_acid_type + '/'
    
    # data reading
    pred_seq = read_data(pred_file_path)
    # feature extraction
    featuresall = seq2features(pred_seq, amino_acid_type)
    # model path
    pnpath = "models/all/" + model_path
    # prediction
    pred_comb = list(test_predict(featuresall, pnpath))
    # results storage
    write_results('GPSD_prediction_results.txt', pred_seq, pred_comb)
    

if __name__ == '__main__':
    main()