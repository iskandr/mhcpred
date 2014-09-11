import numpy as np
import pandas as pd 
from collections import Counter 
import cPickle 

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



def leave_one_out(X_idx, Y_IC50, Y_cat, alleles, 
                  n_iters = 5, 
                  binding_cutoff = 500,
                  output_file_name = "aa_features_cv_results.csv",
                  clf = False):
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

    n_samples = len(Y_IC50)
    
    accuracies = []
    sensitivities = []
    specificities = []
    aucs = []
    
    unique_human_alleles = set(a for a in alleles if a.startswith("HLA"))


    np.random.seed(1)
    results = {}
    output_file = open(output_file_name, 'w')
    output_file.write("Allele,PCC,AUC,Sensitivity,Specificity\n")

    Y_lte = Y_IC50 <= binding_cutoff 

    try:
        for allele in sorted(unique_human_alleles):
            print 
            print ">>>", allele 

            mask = ~np.array([x == allele for x in alleles])
            if (Y_IC50[mask] <= binding_cutoff).std() == 0 or \
                    (Y_IC50[~mask] <= binding_cutoff).std() == 0:
                print "Skipping %s" % allele
                continue 
            
            X_train = X[mask]
            X_test = X[~mask]
            Y_train = Y_IC50[mask]
            Y_lte_train = Y_lte[mask]
            Y_lte_test = Y_lte[~mask]


            if clf:
                model = sklearn.ensemble.RandomForestClassifier(200)
                model.fit(X_train, Y_lte_train)
                pred = model.predict_proba(X_test)[:,1]
                pred_lte = pred >= 0.5
                auc = sklearn.metrics.roc_auc_score(Y_lte_test, pred)
        
            else:
                model = sklearn.ensemble.RandomForestClassifier(200)
                maxval = 20000
                Y_train[Y_train>maxval] = maxval
                Y_train = np.log(Y_train) / np.log(maxval)
                model.fit(X_train, Y_train)
                pred = maxval ** model.predict(X_test)
                pred_lte = pred <= binding_cutoff
                auc = sklearn.metrics.roc_auc_score(Y_lte_test, -pred)

            
            print "Predicted binders fraction", np.mean(pred_lte)
        
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

            output_file.write("%s,%f,%f,%f,%f\n" % \
                (allele,accuracy,auc,sensitivity,specificity))
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
    Y_IC50 = np.load("aa_features_Y_IC50.npy")
    Y_cat = np.load("aa_features_Y_category.npy")

    with open('aa_features_alleles.txt', 'r') as f:
        alleles = [l.strip() for l in f.read().split("\n") if len(l) > 0]
    assert len(X) == len(Y_IC50)
    assert len(X) == len(Y_cat)
    assert len(X.shape) == 2
    assert len(alleles) == len(X)
    mask = (Y_IC50 > 0) & ~np.isinf(Y_IC50) & ~np.isnan(Y_IC50) & (Y_IC50< 10**7)
    X = X[mask]
    Y_IC50 = Y_IC50[mask]
    Y_cat = Y_cat[mask]
    alleles = [alleles[i] for i,b in enumerate(mask) if b]
    

    print "Loaded X.shape = %s" % (X.shape,)
    alleles, (X, Y_IC50, Y_cat) = shuffle(alleles, X, Y_IC50, Y_cat)
    leave_one_out(X, Y_IC50, Y_cat, alleles)