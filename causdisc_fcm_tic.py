# -----------------------------------------------------------------------------
# File    : causdisc_fcm_tic.py
# Created : 2023-07-30
# By      : Alexandre Trilla <alexandre.trilla@alstomgroup.com>
#
# Causal Discovery based on Functional Models
# -----------------------------------------------------------------------------

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mutual_info_score
import os
from os.path import exists
import pandas as pd
import logging
import csv
import pickle
import re
import scipy.stats as stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

from PyRKHSstats import hsic
from PyRKHSstats.kernel_wrapper import KernelWrapper
from sklearn.gaussian_process.kernels import RBF
from PyRKHSstats.combinatorics_utilities import n_permute_m, ordered_combinations, generate_strict_permutations

from statsmodels.stats.diagnostic import lilliefors as lftest


# Challenge dataset: https://www.causality.inf.ethz.ch/cause-effect.php?page=data

# Challenge SUP2: accuracy baseline
# - Chi2/Lin: 0.3850 0.0181
#   CC: 0.2867 0.1612
#   BB: 0.5027 0.1854
#   NN: 0.4300 0.0260 <---------------------
#   MIX: 0.3215 0.0392 <---------------------

# - TIC/Lin: 0.3211 0.0136
#   CC: 0.3500 0.1329 <----------------------
#   BB: 0.5336 0.1375 <----------------------
#   NN: 0.3683 0.0211
#   MIX: 0.2456 0.0219



def TestDNNReg():
    """ Eval performance of DNN for regression of sinewave.

    Returns:
        Print RMS and plot curves.

    """
    x = 0.05*np.random.randn(1000)
    #t = np.linspace(0, 2*np.pi, 1000)
    #x = x + np.sin(t)
    #t = np.linspace(-2, 2, 1000)
    #x = x + t**2
    #t = np.linspace(0, 10, 1000)
    #x = x + 1.0 - np.exp(-t)
    t = np.linspace(-1.5, 1.5, 1000)
    x = x + t**3
    X = np.reshape(t, (len(t), 1))
    Y = np.reshape(x, (len(x), 1))
    Y = Y.flatten()
    regf = MLPReg()
    regYX = regf.fit(X, Y)
    pred = regYX.predict(X)
    rYX = Y - pred
    resid = rYX.flatten()
    rms = np.sqrt(np.dot(resid,resid))
    print("RMS: " + str(rms))
    plt.plot(t,x,'bo')
    plt.plot(t, pred, 'r')
    plt.show()


def MLPReg():
    """ Fine-tuned MLP regressor.

    Args:
        x (num array): First var.
        y (num array): Second var.

    Returns:
        mlp (obj): MLP function.

    """
    mlp = MLPRegressor(hidden_layer_sizes=(16,16), max_iter=2000)
    return mlp


def Chi2(x, y, bins=10):
    """ Pearson's Chi2 indep test.

    Args:
        x (num array): First var.
        y (num array): Second var.

    Returns:
        pval (real): p-value of the test.

    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    c_xy = np.histogram2d(x, y, bins)[0]
    # Remove all zero rows and/or columns
    zcols = np.sum(c_xy, 0)
    redcm = c_xy[:,zcols!=0]
    zrows = np.sum(redcm, 1)
    redcm = redcm[zrows!=0,:]
    g, p, dof, expected = chi2_contingency(redcm)
    return p


def MI(x, y):
    """ Calculate Mutual Info.

    Args:
        x (num array): First var.
        y (num array): Second var.

    Returns:
        mi (real): Mutual Info.

    """
    bins = 10
    x = x.flatten()
    y = y.flatten()
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def HSIC(x, y):
    """ Calculate the Hilbert-Schmidt Independence Criterion test.

    Args:
        x (num array): First var.
        y (num array): Second var.

    Returns:
        kpi (real): p-val.

    """
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    length_scale = 0.1 ** (-0.5)
    kernel_x = KernelWrapper(RBF(length_scale=length_scale))
    kernel_y = KernelWrapper(RBF(length_scale=length_scale))
    gperm = generate_strict_permutations(
            indices_to_permute=list(range(x.shape[0])),
            nb_permutations_wanted=1000)
    test = hsic.perform_permutation_hsic_independence_testing(
            x,y,kernel_x,kernel_y,permutations=gperm, test_level=0.05)
    res = list(test.values())[0]
    pval = 0.0
    if not res:
        pval = 1.0
    return pval


def TIC(x, y):
    """ Calculate Total Information Coefficient indep test.

    Args:
        x (num array): First var.
        y (num array): Second var.

    Returns:
        tic (real): TICePVal

    """
    return MIC(x,y)[0]


def MIC(x, y):
    """ Calculate Maximal Information Coefficient. Linux env.

    Args:
        x (num array): First var.
        y (num array): Second var.

    Returns:
        tic (real): TICePVal
        mic (real): MICe

    """
    tic = -1
    mic = -1
    # prep data
    os.system("mkdir foo")
    os.system("mkdir foores")
    dlen = len(x)
    idx = np.arange(dlen, dtype=int) + 1
    x = x.flatten()
    y = y.flatten()
    mat = np.array([idx.astype(int),x,y])
    df = pd.DataFrame(mat, index=['','X','Y'])
    df.to_csv("foo/data.tsv", header=False, sep='\t')
    # calc mic
    os.system("mictools null foo/data.tsv foores/null_dist.txt")
    os.system("mictools pval foo/data.tsv foores/null_dist.txt foores")
    os.system("mictools adjust foores/pval.txt foores")
    os.system("mictools strength foo/data.tsv foores/pval_adj.txt foores/strength.txt")
    # check result TICe
    f = open("foores/pval.txt", "r")
    head = f.readline()
    resvals = f.readline()
    if (len(resvals)>10):
        vals = resvals.split('\t')
        tic = vals[2]
        tic = float(tic[:-1])
    # check result MICe
    f = open("foores/strength.txt", "r")
    head = f.readline()
    resvals = f.readline()
    if (len(resvals)>10):
        vals = resvals.split('\t')
        mic = vals[6]
        mic = float(mic[:-1])
    os.system("rm -rf foo")
    os.system("rm -rf foores")
    return tic, mic


def GenData(numdat=1000):
    """ Generate synthetic dataset of N samples using uniform random vars.

    Returns:
        x_to_y (matrix): X -> Y (N,2)
        x_indep_y (matrix): X _||_ Y (N,2)
        x_conf_y (matrix): X <-> Y (N,2)

    """
    Z = np.random.random_sample(numdat)
    X_ind = np.random.random_sample(numdat)
    Y_ind = np.random.random_sample(numdat)
    Y_dep = X_ind + Z
    X_conf = X_ind + Z
    Y_conf = Y_ind + Z
    #
    caus = np.array([X_ind, Y_dep])
    caus = caus.transpose()
    inde = np.array([X_ind, Y_ind])
    inde = inde.transpose()
    conf = np.array([X_conf, Y_conf])
    conf = conf.transpose()
    return caus,inde,conf


def FCMReg(X, Y, uit, regf):
    """ Discover causality assuming linear link between X and Y.

    Args:
        X (num array): Hypotehsis cause.
        Y (num array): Hypotehsis effect.
        uit (fun): Unconditional independence test function.
        regf (fun): Function to compute the FCM regression.

    Returns:
        kpi (real): Indep test (p-value) between lin reg residual and 
            hypothesis cause.

    """
    X = np.reshape(X, (len(X), 1))
    Y = np.reshape(Y, (len(Y), 1))
    Y = Y.flatten()
    regYX = regf().fit(X, Y)
    rYX = Y - regYX.predict(X)
    resid = rYX.flatten()
    return uit(X, resid)


def CDFCM(A, B, uit, regf):
    """ Causal Discovery using FCM pairwise method.

    Args:
        A (num array): Hypotehsis cause.
        B (num array): Hypotehsis effect.
        uit (fun): Unconditional independence test function.
        regf (fun): Function to compute the FCM regression.

    Returns:
        Int. 1 for A->B; 2 for A<-B; 3 for A-B; 4 for A|B

    """
    toret = 1
    ci = 0.05
    AB = FCMReg(A, B, uit, regf)
    BA = FCMReg(B, A, uit, regf)
    if (AB > ci) and (BA < ci):
        pass
    if (AB < ci) and (BA > ci):
        toret = 2
    if (AB > ci) and (BA > ci):
        toret = 4
    if (AB < ci) and (BA < ci):
        toret = 3
    return toret


def CD_Batch(dataset, path, uit, regf):
    """ Apply Causal Discovery on a dataset.

    Args:
        dataset (list(matrix)): List of instances, of values for 2 variables.
        path (str): Path to folder.
        uit (fun): Unconditional independence test function.
        regf (fun): Function to compute the FCM regression.

    Returns:
        caus (list(int)): Array of M values (1-4)

    """
    caus = []
    fnam = path + "/batch.log"
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        filename=fnam, level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Batch started")
    for datanum in range(len(dataset)):
        data = dataset[datanum]
        causdisc = CDFCM(data[:,0], data[:,1], uit, regf)
        logging.info(str(datanum) + " - " + str(causdisc))
        caus.append(causdisc)
    return caus


def _StitchBatch(fnam):
    """ Convert a manually stitched batch file into a pred.pickle file.

    Args:
        fnam (str): Name of the file. "INFO:root:# - R", where R is result.

    Returns:
        Created a valid pred.pickle file containing a pickled list of ints.

    """
    data = []
    f = open(fnam, 'r')
    l = f.readline()
    while (l != ''):
        if (len(l) > 2):
            chunk = l.split("-")
            data.append(int(chunk[1][1]))
        l = f.readline()
    f.close()
    f = open("pred.pickle", 'wb')
    pickle.dump(data, f)
    f.close()


def TestCD():
    """ Test CD using synthetic data, uniform distrib.

    Returns:
        Logs results into testcd.log file.

    """
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        filename='testcd.log', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Test CD started")
    #
    caus,inde,conf = GenData()
    #
    logging.info("##########   X->Y")
    logging.info("Lin + Chi2: " + 
                 str(CDFCM(caus[:,0], caus[:,1], Chi2, LinearRegression)))
    #logging.info("Lin + HSIC: " + 
    #             str(CDFCM(caus[:,0], caus[:,1], HSIC, LinearRegression)))
    #
    logging.info("##########   X_|_Y")
    logging.info("Lin + Chi2: " + 
                 str(CDFCM(inde[:,0], inde[:,1], Chi2, LinearRegression)))
    #logging.info("Lin + HSIC: " + 
    #             str(CDFCM(inde[:,0], inde[:,1], HSIC, LinearRegression)))
    #
    logging.info("##########   X<->Y")
    logging.info("Lin + Chi2: " + 
                 str(CDFCM(conf[:,0], conf[:,1], Chi2, LinearRegression)))
    #logging.info("Lin + HSIC: " + 
    #             str(CDFCM(conf[:,0], conf[:,1], HSIC, LinearRegression)))


def PlotSynth():
    """ Test using synthetic data, uniform distrib, MI and Chi2.

    Returns:
        Intermediate processes as pickles.
        MI noise Plot and chart of p-vals.

    """
    #
    mi_c = []
    mi_i = []
    mi_co = []
    mi_a = []
    c2_c_c = []
    c2_c_a = []
    c2_i_c = []
    c2_i_a = []
    c2_co_c = []
    c2_co_a = []
    for i in range(300):
        print("Epoch " + str(i))
        caus,inde,conf = GenData(numdat=2000)
        mi_c.append(FCMReg(caus[:,0], caus[:,1], MI, LinearRegression))
        mi_i.append(FCMReg(inde[:,0], inde[:,1], MI, LinearRegression))
        mi_co.append(FCMReg(conf[:,0], conf[:,1], MI, LinearRegression))
        mi_a.append(FCMReg(caus[:,1], caus[:,0], MI, LinearRegression))
        #
        c2_c_c.append(FCMReg(caus[:,0], caus[:,1], Chi2, LinearRegression))
        c2_c_a.append(FCMReg(caus[:,1], caus[:,0], Chi2, LinearRegression))
        c2_i_c.append(FCMReg(inde[:,0], inde[:,1], Chi2, LinearRegression))
        c2_i_a.append(FCMReg(inde[:,1], inde[:,0], Chi2, LinearRegression))
        c2_co_c.append(FCMReg(conf[:,0], conf[:,1], Chi2, LinearRegression))
        c2_co_a.append(FCMReg(conf[:,1], conf[:,0], Chi2, LinearRegression))

#    plt.figure()
#    plt.hist(mi_c, bins=10, color="#FF0000", alpha=0.5, label="Causal")
#    plt.hist(mi_a, bins=10, color="#00FF00", alpha=0.5, label="Anticausal")
#    plt.hist(mi_i, bins=10, color="#FFFF00", alpha=0.5, label="Independent")
#    plt.hist(mi_co, bins=10, color="#0000FF", alpha=0.5, label="Confounded")
#    plt.xlabel("Mutual Information")
#    plt.ylabel("Histogram")

#    plt.figure()
#    binwidth = 0.004
#    lbins = np.arange(0, 0.25 + binwidth, binwidth)
#    plt.hist(mi_c, bins=lbins, color="#FF0000", alpha=0.5, histtype='stepfilled', density=True, label="Causal")
#    plt.hist(mi_a, bins=lbins, color="#00FF00", alpha=0.5, histtype='stepfilled', density=True, label="Anticausal")
#    plt.hist(mi_i, bins=lbins, color="#FFFF00", alpha=0.5, histtype='stepfilled', density=True, label="Independent")
#    plt.hist(mi_co, bins=lbins, color="#0000FF", alpha=0.5, histtype='stepfilled', density=True, label="Confounded")
#    plt.xlabel("Mutual Information")
#    plt.ylabel("Histogram")
#    plt.legend(loc='upper right')
#    plt.savefig("mi_hist.pdf", bbox_inches="tight")




    plt.figure()
    binwidth = 0.002
    lbins = np.arange(0, 0.25 + binwidth, binwidth)

    smooth = 2

    causal_density = stats.gaussian_kde(mi_c)
    causal_density.covariance_factor = lambda : causal_density.factor*smooth
    causal_density._compute_covariance()
    plt.plot(lbins, causal_density(lbins), color="#FF0000", linewidth=1, label="Causal")
    plt.fill_between(lbins,causal_density(lbins), color="#FF0000", alpha=0.5)

    acausal_density = stats.gaussian_kde(mi_a)
    acausal_density.covariance_factor = lambda : acausal_density.factor*smooth
    acausal_density._compute_covariance()
    plt.plot(lbins, acausal_density(lbins), color="#00FF00", linewidth=1, label="Anticausal")
    plt.fill_between(lbins,acausal_density(lbins), color="#00FF00", alpha=0.5)

    ind_density = stats.gaussian_kde(mi_i)
    ind_density.covariance_factor = lambda : ind_density.factor*smooth
    ind_density._compute_covariance()
    plt.plot(lbins, ind_density(lbins), color="#FFFF00", linewidth=1, label="Independent")
    plt.fill_between(lbins,ind_density(lbins), color="#FFFF00", alpha=0.5)

    conf_density = stats.gaussian_kde(mi_co)
    conf_density.covariance_factor = lambda : conf_density.factor*smooth
    conf_density._compute_covariance()
    plt.plot(lbins, conf_density(lbins), color="#0000FF", linewidth=1, label="Confounded")
    plt.fill_between(lbins,conf_density(lbins), color="#0000FF", alpha=0.5)

    plt.xlabel("Mutual Information")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.savefig("mi_dens.pdf", bbox_inches="tight")

    #
    print("C/X->Y  " + str(np.mean(c2_c_c)))
    print("AC/X->Y  " + str(np.mean(c2_c_a)))
    print("C/X|Y  " + str(np.mean(c2_i_c)))
    print("AC/X|Y  " + str(np.mean(c2_i_a)))
    print("C/X-Y  " + str(np.mean(c2_co_c)))
    print("AC/X-Y  " + str(np.mean(c2_co_a)))


def TestSynth(fkpi):
    """ Test using synthetic data, uniform distrib.

    Args:
        fkpi (function): Indep eval function.

    Returns:
        Logs results into testsynth.log file.

    """
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        filename='testsynth.log', level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Test synth started")
    #
    caus,inde,conf = GenData()
    #
    logging.info("##########   X->Y")
    logging.info(str(DiscLinFCM(caus[:,0], caus[:,1], fkpi)))
    logging.info("##########   Y->X")
    logging.info(str(DiscLinFCM(caus[:,1], caus[:,0], fkpi)))
    logging.info("Lin + Chi2: " + 
                 str(CDFCM(caus[:,0], caus[:,1], Chi2, LinearRegression)))
    logging.info("Lin + HSIC: " + 
                 str(CDFCM(caus[:,0], caus[:,1], HSIC, LinearRegression)))
    #
    logging.info("##########   X _||_ Y")
    logging.info(str(DiscLinFCM(inde[:,0], inde[:,1], fkpi)))
    logging.info("##########   Y _||_ X")
    logging.info(str(DiscLinFCM(inde[:,1], inde[:,0], fkpi)))
    #
    logging.info("##########   X <-> Y")
    logging.info(str(DiscLinFCM(conf[:,0], conf[:,1], fkpi)))
    logging.info("##########   Y <-> X")
    logging.info(str(DiscLinFCM(conf[:,1], conf[:,0], fkpi)))


def LoadChallenge(path, numFiles=-1):
    """ Load Challenge dataset. Causal/non-causal/indep/confounded classes.

        Folder contains:
            trainX.txt -> (tab,space,...)-separated X Y vars.
            CEdata_train_target -> line head: explain... 
                      comma-separated fields, last one is gound truth (1-4)

    Args:
        path (str): Path to folder.
        numFiles (int): Number of files.

    Returns:
        data (list(matrix)): List of instances, of values for 2 variables.
        gt (list(int)): Array of M values. Ground truth (1: ->, 2: <-) 

    """
    def getData(fnam):
        data = []
        f = open(fnam, 'r')
        l = f.readline()
        while (l != ''):
            if (len(l) > 2):
                l = re.sub("\t"," ", l)
                l = re.sub("\s+"," ", l)
                l = re.sub("^\s","", l)
                chunk = l.split(" ")
                d = np.zeros((2))
                d[0] = float(chunk[0])
                d[1] = float(chunk[1])
                data.append(d)
            l = f.readline()
        f.close()
        data = np.array(data)
        return data

    def getGT(fnam):
        data = []
        f = open(fnam, 'r')
        l = f.readline()
        l = f.readline()
        while (l != ''):
            if (len(l) > 2):
                chunk = l.split(",")
                data.append(int(chunk[2][0]))
            l = f.readline()
        f.close()
        return data

    data = []
    gt = []
    # Check if it has been processed before
    fnam = path + "/dataset.pickle"
    if exists(fnam):
        f = open(fnam, 'rb')
        data,gt = pickle.load(f)
        f.close()
    else:
        # iterate from 1 to numFiles
        for i in range(1,numFiles+1):
            print(i)
            # load values into mat, append to data
            number = str(i)
            data.append(getData(path + "/train" + number + ".txt"))
        #
        gt = getGT(path + "/CEdata_train_target.csv")
        f = open(fnam, 'wb')
        pickle.dump((data,gt), f)
        f.close()
    return data, gt


def LoadTuebingen(path):
    """ Load Tuebingen dataset. Causal/non-causal classification.

        Folder contains:
            pair0XXX.txt -> (tab,space,...)-separated X Y vars.
            README -> line head: pair0XXX 
                      tab-separated fields, last one is gound truth (-> or <-)
                      (pair0093 has to be manually edited)

    Args:
        path (str): Path to folder.

    Returns:
        data (list(matrix)): List of instances, of values for 2 variables.
        gt (list(int)): Array of M values. Ground truth (1: ->, 2: <-) 

    """
    def padNumber(i):
        itos = ''
        si = str(i)
        if (len(si) == 1):
            itos = '00' + si
        if (len(si) == 2):
            itos = '0' + si
        if (len(si) == 3):
            itos = si
        return itos

    def nextPair(f):
        thend = False
        toret = []
        while (not thend):
            l = f.readline()
            if (l.startswith('pair')):
                toret = l
                thend = True
            if (l == ''):
                thend = True
        return toret
    
    def getData(fnam):
        data = []
        f = open(fnam, 'r')
        l = f.readline()
        while (l != ''):
            if (len(l) > 2):
                l = re.sub("\t"," ", l)
                l = re.sub("\s+"," ", l)
                l = re.sub("^\s","", l)
                chunk = l.split(" ")
                d = np.zeros((2))
                d[0] = float(chunk[0])
                d[1] = float(chunk[1])
                data.append(d)
            l = f.readline()
        f.close()
        data = np.array(data)
        return data

    data = []
    gt = []
    # Check if it has been processed before
    fnam = path + "/dataset.pickle"
    if exists(fnam):
        f = open(fnam, 'rb')
        data,gt = pickle.load(f)
        f.close()
    else:
        # iterate from 1 to 108
        fgt = open(path + "/README", 'r')
        for i in range(1,109):
            print(i)
            # load values into mat, append to data
            number = padNumber(i)
            data.append(getData(path + "/pair0" + number + ".txt"))
            # get GT from README, append to gt
            nextgt = nextPair(fgt)
            if len(nextgt) > 0:
                if (nextgt.startswith("pair0" + number)):
                    if (nextgt.count('->') > 0):
                        gt.append(1)
                    else:
                        gt.append(2)
                else:
                    print("MISMATCH ERR " + number)
        fgt.close()
        f = open(fnam, 'wb')
        pickle.dump((data,gt), f)
        f.close()
    return data, gt


def FoldAccuracy(gt, pred):
    """ Estimate average accuracy on a 10-fold validation.

    Args:
        gt (list(int)): Array of values. Ground truth.
        pred (list(int)): Array of values. Predictions.

    Returns:
        mu (float): Predicted mean accuracy.
        sigma (float): Predicted stdev accuracy.

    """
    def Accuracy(x,y):
        match = x==y
        mu = np.mean(match)
        return mu

    def FoldIdx(numdata, numfolds):
        step = int(numdata/numfolds)
        arr = np.linspace(0,numdata,numfolds+1)
        toret = []
        for i in arr:
            toret.append(int(i))
        return toret

    gt = np.array(gt)
    pred = np.array(pred)
    accs = []
    idx = FoldIdx(len(gt), 10)
    for fold in range(1,len(idx)):
        m = Accuracy(gt[idx[fold-1]:idx[fold]], pred[idx[fold-1]:idx[fold]])
        accs.append(m)
    # check Normality
    ksstat_leave,pvalue_leave = lftest(accs)
    if pvalue_leave < 0.05:
        print("Reject Normality")
    return np.mean(accs), np.std(accs)


def getDataType(path):
    """ Get data types at instance level for Challenge dataset.

    Args:
        path (str): Path to dataset.

    Returns:
        cdt (list(str)): Data types AB, where: c, b, n.

    """
    datatype = []
    f = open(path + "/CEdata_train_publicinfo.csv", 'r')
    l = f.readline()
    l = f.readline()
    while (l != ''):
        if (len(l) > 2):
            chunk = l.split(",")
            styp = ""
            if chunk[1].startswith("B"):
                styp = styp + "b"
            if chunk[1].startswith("C"):
                styp = styp + "c"
            if chunk[1].startswith("N"):
                styp = styp + "n"
            if chunk[2].startswith("B"):
                styp = styp + "b"
            if chunk[2].startswith("C"):
                styp = styp + "c"
            if chunk[2].startswith("N"):
                styp = styp + "n"
            datatype.append(styp)
        l = f.readline()
    f.close()
    return datatype


def RunDataset(path, loader, bline_mu, bline_std, uit, regf, chdatype="none"):
    """ Run experiment on a dataset.

    The Challenge data types can be introduced for fine-grained eval purposes.

    Args:
        path (str): Path to dataset.
        loader (func): Function to load the data.
        bline_mu (float): Baseline mean accuracy.
        bline_std (float): Baseline stdev accuracy.
        uit (fun): Unconditional independence test function.
        regf (fun): Function to compute the FCM regression.
        chdatype (str): Challenge data type. none, cc, bb, nn, mix.

    Returns:
        mu (float): Predicted mean accuracy.
        sigma (float): Predicted stdev accuracy.
        pval (float): t-test pval.

    """

    data,gt = loader(path)
    #
    fresnam = path + "/pred.pickle"
    if exists(fresnam):
        f = open(fresnam, 'rb')
        pred = pickle.load(f)
        f.close()
    else:
        pred = CD_Batch(data, path, uit, regf)
        f = open(fresnam, 'wb')
        pickle.dump(pred, f)
        f.close()
    #
    # select data type for performance eval
    filtgt = []
    filtpred = []
    if chdatype != "none":
        cdt = getDataType(path)
        for t in range(len(cdt)):
            if cdt[t] == chdatype:
                filtgt.append(gt[t])
                filtpred.append(pred[t])
            elif chdatype == "mix":
                if cdt[t][0] != cdt[t][1]:
                    filtgt.append(gt[t])
                    filtpred.append(pred[t])
    else:
        filtgt = gt
        filtpred = pred
    mu,sigma = FoldAccuracy(filtgt, filtpred)
    #
    baseline = np.random.randn(500)*bline_std + bline_mu
    improv = np.random.randn(500)*sigma + mu
    t,pval = stats.ttest_ind(baseline, improv)
    #
    return mu,sigma,pval


def distMixGauss(weights, mus, stds):
    """ Distribution of a weighted mixture of Gaussians.

    Args:
        weights (list(float)): List of weights (add to 1).
        mus (list(float)): List of means.
        stds (list(float)): List of stdevs.

    Returns:
        emp_mu (float): Empirical Mean of the mixture.
        emp_std (float): Empirical Std of the mixture.
        theo_mu (float): Theoretical Mean of the mixture.
        theo_std (float): Theoretical Std of the mixture.

    """
    # empirical
    N = 10000
    sample = np.zeros(N)
    cumul = 0
    for i in range(len(weights)):
        part = int(N*weights[i])
        if (i == len(weights)-1):
            part = N - cumul
        sample[cumul:(cumul+part)] = np.random.randn(part)*stds[i] + mus[i]
        cumul = cumul + part
    emp_mu = np.mean(sample)
    emp_std = np.std(sample)
    # theoretical
    w = np.array(weights)
    m = np.array(mus)
    s = np.array(stds)
    theo_mu = np.dot(w, m)
    theo_std = np.sqrt(np.dot(w, s**2 + m**2) - theo_mu**2)
    #
    return emp_mu, emp_std, theo_mu, theo_std


def statSignifDiff(mu1, std1, mu2, std2):
    """ Check if the differece between Normal distribs is stat signif.

    Args:
        mu1 (float): Baseline mean accuracy.
        std1 (float): Baseline stdev accuracy.
        mu2 (float): Baseline mean accuracy.
        std2 (float): Baseline stdev accuracy.

    Returns:
        pval (float): t-test pval.

    """
    baseline = np.random.randn(500)*std1 + mu1
    improv = np.random.randn(500)*std2 + mu2
    t,pval = stats.ttest_ind(baseline, improv)
    return pval


def PlotChallengeLength():
    """ Plot the distribution of record lengths.

    Returns:
        Plot histogram of record lenth.

    """
    data,gt = LoadChallenge("PATH\\SUP2data_split", 5989)
    dtype = getDataType("PATH\\SUP2data_split")
    #
    lens_cc = []
    lens_bb = []
    lens_nn = []
    lens_mx = []
    for i in range(len(data)):
        leni = data[i].shape[0]
        typi = dtype[i]
        if typi == 'cc':
            lens_cc.append(leni)
        elif typi == 'bb':
            lens_bb.append(leni)
        elif typi == 'nn':
            lens_nn.append(leni)
        else:
            lens_mx.append(leni)
    plt.figure()
    binwidth = 200
    lbins = range(0, 8000 + binwidth, binwidth)
    plt.hist(lens_nn, bins=lbins, color="#FFFF00", alpha=1, label="Numerical")
    plt.hist(lens_mx, bins=lbins, color="#0000FF", alpha=1, label="Mixed")
    plt.hist(lens_bb, bins=lbins, color="#00FF00", alpha=1, label="Binary")
    plt.hist(lens_cc, bins=lbins, color="#FF0000", alpha=1, label="Categorical")
    plt.xlabel("Sample Size [Record Length]")
    plt.ylabel("Histogram")
    plt.legend(loc='upper right')
    plt.savefig("sampsize_hist.pdf", bbox_inches="tight")
    print("CC: " + str(np.median(lens_cc)))
    print("BB: " + str(np.median(lens_bb)))
    print("NN: " + str(np.median(lens_nn)))
    print("MX: " + str(np.median(lens_mx)))


if __name__ == '__main__':
    #TestSynth(MIC)
    #
    #TestCD()
    #
    #
    #LoadChallenge("PATH\\SUP2data_split", 5989)
    #
    #_StitchBatch("results_chal_sup2_tic_dnn.txt")
    #
    #mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, TIC, LinearRegression)
    #
    #mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, Chi2, LinearRegression)
    #mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, Chi2, MLPReg)
    #mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, HSIC, LinearRegression)
    #
    mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, "", "", chdatype="cc")
    print("cc: ", mu,sigma,pval)
    mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, "", "", chdatype="bb")
    print("bb: ", mu,sigma,pval)
    mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, "", "", chdatype="nn")
    print("nn: ", mu,sigma,pval)
    mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, "", "", chdatype="mix")
    print("mix: ", mu,sigma,pval)
    mu,sigma,pval = RunDataset("PATH\\SUP2data_split", LoadChallenge, 0.67, 0.02, "", "")
    print("total: ", mu,sigma,pval)
    #
    #TestDNNReg()
    #PlotSynth()
    #
    #print(str(statSignifDiff(0.2867, 0.1612, 0.3500, 0.1329)))
    #print(str(statSignifDiff(0.5027, 0.1854, 0.5336, 0.1375)))
    #print(str(statSignifDiff(0.4300, 0.0260, 0.3683, 0.0211)))
    #print(str(statSignifDiff(0.3215, 0.0392, 0.2456, 0.0219)))
    #print(str(statSignifDiff(0.3850, 0.0181, 0.3211, 0.0136)))
    # ACE
    #emp_mu, emp_std, theo_mu1, theo_std1 = distMixGauss([0.5,0.5],[0.3211,0.3850],[0.0136,0.0181])
    #print("Empirical", emp_mu, emp_std)
    #print("Theoretical", theo_mu1, theo_std1)
    #emp_mu, emp_std, theo_mu2, theo_std2 = distMixGauss([.0154,.0170,.5630,.4046],[0.3500,0.5336,0.4300,0.3215],[0.1329,0.1375,0.0260,0.0392])
    #print("Empirical", emp_mu, emp_std)
    #print("Theoretical", theo_mu2, theo_std2)
    #
    #print("ACE: ", theo_mu2 - theo_mu1)
    #print("Stat signif: ", (statSignifDiff(theo_mu2, theo_std2, theo_mu1, theo_std1)))
    #
    #PlotChallengeLength()


