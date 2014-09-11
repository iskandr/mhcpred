import numpy as np
import pandas as pd 
from collections import Counter 
import cPickle 
import math

import pmbec 
from epitopes import amino_acid 
from parakeet import jit 
import scipy.weave 
import sklearn.linear_model
from sklearn.linear_model import LassoCV
import sklearn.svm 
import sklearn.ensemble
import sklearn.decomposition 

from log_linear_regression import LogLinearRegression
from two_pass_regressor import TwoPassRegressor
from selective_regressor import SelectiveRegressor 
from generate_training_data import generate_training_data
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS



import math 

import numpy as np 
import pandas as pd 

import sklearn.linear_model
import sklearn.metrics 

import epitopes.amino_acid
from parakeet import jit 
from censored_regression import CensoredLasso, CensoredRidge
        

import pmbec 
from log_linear_regression import LogLinearRegression
from two_pass_regressor import TwoPassRegressor
from selective_regressor import SelectiveRegressor 
from generate_training_data import generate_training_data
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS


def leave_one_out(
        X, X_pep, 
        Y_IC50, Y_cat, alleles, 
        binding_cutoff = 500,
        normalize = False,
        output_file_name = "aa_features_cv_results.csv"):
    """
    X_idx : 2d array of integers with shape = (n_samples, n_features) 
        Elements are indirect references to elements of the 
        feature encoding matrix
    
    Y_IC50 : 1d array of floats with shape = (n_samples,) 
        target IC50 values

    Y_cat : 1d array of boolean elements with shape = (n_samples,)

    alleles : str list 

    cutoff : int 
        Binding affinity threshold below which a peptide is considered a binder

    n_iters : int 
        Number of training iterations to refine feature matrix 
    """


    #pca = sklearn.decomposition.RandomizedPCA(100)
    std = X.std(axis=0)
    bad_mask = std < 0.01
    print "Dropping low-variance columns:", bad_mask.nonzero()
    X = X[:, ~bad_mask]

    n_samples = len(X)
    accuracies = []
    sensitivities = []
    specificities = []
    aucs = []
    counts = []
    
    unique_human_alleles = set(a for a in alleles if a.startswith("HLA"))



    results = {}
    output_file = open(output_file_name, 'w')
    output_file.write("Allele,Count,AUC,Accuracy,Sensitivity,Specificity\n")


    Y_lte = Y_cat
    maxval = 20000
    Y_IC50[Y_IC50>maxval] = maxval
    
    try:
        for allele in sorted(unique_human_alleles):
             

            allele_mask = np.array([x == allele for x in alleles])
            not_allele_mask = ~allele_mask
            n = allele_mask.sum()
            print 
            print ">>>", allele, "(n = %d)" % n
            counts.append(n)

            X_train = X[not_allele_mask]
            X_train_pep = X_pep[not_allele_mask]
            X_test = X[allele_mask]
            X_test_pep = X_pep[allele_mask]
            Y_train = Y_cat[not_allele_mask]
            Y_test = Y_cat[allele_mask]
            Y_lte_train = Y_train > 0
            Y_lte_test = Y_test > 0
            if Y_lte_train.mean() in (0, 1) or Y_lte_test.mean() in (0,1):
                print "Skipping due to lack of variance in output"
                continue 

            if normalize:
                Xm = X_train.mean(axis=0)
                X_train -= Xm
                X_test -= Xm 
                Xs = X_train.std(axis=0)
                X_train /= Xs 
                X_test /= Xs
            
            model = sklearn.ensemble.RandomForestClassifier(300)
            if len(X_test) >= 25:
                solo_aucs = sklearn.cross_validation.cross_val_score(
                    model, 
                    X_test_pep, 
                    Y_lte_test, 
                    scoring="roc_auc") 
                solo_auc = np.mean(solo_aucs)
                print "--- Solo CV for %s: %0.4f" % (allele, solo_auc)
            model.fit(X_train, Y_train)
            probs = model.predict_proba(X_test)
            pred = probs[:, -1] + probs[:, -2]
            pred_lte =  pred > 0.5
            auc = sklearn.metrics.roc_auc_score(Y_lte_test, pred)
            
            
            pred_gt = ~pred_lte 
            actual_gt = ~Y_lte_test
            correct = (pred_lte & Y_lte_test) | (pred_gt & actual_gt)
            accuracy = np.mean(correct)
            sensitivity = np.mean(pred_lte[Y_lte_test])
            specificity = np.mean(Y_lte_test[pred_lte])

        
            print "--- %d binders of %d identified" % (
                (correct & pred_lte).sum(), Y_lte_test.sum()
            )
            print "--- accuracy", accuracy 
            print "--- sensitivity", sensitivity 
            print "--- specificity", specificity
            print "--- AUC", auc 

            output_file.write("%s,%d,%f,%f,%f,%f\n" % \
                (allele,n,auc,accuracy,sensitivity,specificity))
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)
            aucs.append(auc)
            results[allele] = (accuracy, auc)
    except KeyboardInterrupt:
        print "KeyboardInterrupt"
        pass 
    output_file.close()
    print "Results:"
    for allele in sorted(results.keys()):
        (acc, auc) = results[allele]
        print "%s Accuracy %0.5f AUC %0.5f" % (allele, acc, auc)
    print "Overall AUC", np.median(aucs)
    print "Overall CV sensitivity =", np.median(sensitivities)
    print "Overall CV specificity =", np.median(specificities)
    print "Overall CV accuracy =", np.median(accuracies)
        



def shuffle(alleles, *arrays):
    n = len(alleles)
    indices = np.arange(n)
    np.random.shuffle(indices)
    arrays = [x[indices] for x in arrays]
    alleles = [alleles[i] for i in indices]
    return alleles, arrays


if __name__ == "__main__":
    
    X = np.load("aa_features_X.npy")

    X_pep = np.load("aa_features_X_pep.npy")
    Y_IC50 = np.load("aa_features_Y_IC50.npy")
    Y_cat = np.load("aa_features_Y_category.npy")

    with open('aa_features_alleles.txt', 'r') as f:
        alleles = [l.strip() for l in f.read().split("\n") if len(l) > 0]
    assert len(X) == len(Y_IC50)
    assert len(X) == len(Y_cat)
    assert len(X.shape) == 2
    assert len(alleles) == len(X)
    
    

    print "Loaded X.shape = %s" % (X.shape,)
    np.random.seed(1)
    alleles, (X, X_pep, Y_IC50, Y_cat) = shuffle(
        alleles, X, X_pep, Y_IC50, Y_cat)
    leave_one_out(X, X_pep, Y_IC50, Y_cat, alleles)