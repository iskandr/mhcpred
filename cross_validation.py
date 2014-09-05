
import numpy as np 
import pandas as pd 

import sklearn.linear_model
import sklearn.metrics 

from parakeet import jit 
from censored_regression import CensoredLasso, CensoredRidge
        

import pmbec 
from log_linear_regression import LogLinearRegression
from two_pass_regressor import TwoPassRegressor
from selective_regressor import SelectiveRegressor 
from generate_training_data import generate_training_data
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS


def feature_dictionary_to_vector(dictionary):
    """
    Takes a dictionary mapping from amino acid letter pairs (e.g. "AC"), 
    re-encodes the keys as indices
    """
    vec = [None] * len(AMINO_ACID_PAIRS)
    for letter_pair, value in dictionary.iteritems():
        idx = AMINO_ACID_PAIR_POSITIONS[letter_pair]
        vec[idx] = value
    assert all(vi is not None for vi in vec) 
    return np.array(vec)

@jit 
def encode_inputs(X_pair_indices, pairwise_feature_vec):
    """
    X_pair_indices : 2d array
        Indices of amino acid combinations for each (peptide, allele pseudosequence) entry
    pairwise_features : dict
        Maps from AA pair indices to continuous values 
    """ 
    n_samples, n_dims = X_pair_indices.shape 
    n_pairwise_features = len(pairwise_feature_vec)
    X_encoded = np.zeros((n_samples, n_dims), dtype=float)
    for row_idx, x in enumerate(X_pair_indices):
        X_encoded[row_idx, :] = pairwise_feature_vec[x]
    return X_encoded

@jit 
def encode_pairwise_coefficients(X_idx, model_weights):
    n_samples, n_position_pairs = X_idx.shape 
    n_amino_acid_pairs = 20 * 20
    coeffs = np.zeros((n_samples, n_amino_acid_pairs), dtype=float)
    for row_idx, x in enumerate(X_idx):
        for col_idx, xi  in enumerate(x):
            coeffs[row_idx, xi] += model_weights[col_idx]
    return coeffs

def estimate_pairwise_features(X_idx, model_weights, Y, 
                                censor_cutoff = 2000, 
                                log_targets = False, 
                                regularization_weight = 1):
    """
    Estimators:
    - ridge : Ridge Regression for all IC50 values

    - ...otherwise: minimize L1 norm s.t. IC50 <= 2000nM have exact same values, non-binders have predict IC50 > 2000nM

    cutoff : float 
        Binding affinity below which we want exact match of predicted values 
    """
    if log_targets:
        Y = np.log(Y) 
        censor_cutoff = np.log(censor_cutoff)

    assert len(model_weights) == X_idx.shape[1]
    assert (X_idx.max() < 20 * 20), X_idx.max()
    C = encode_pairwise_coefficients(X_idx, model_weights)

    model = CensoredLasso(regularization_weight = regularization_weight, verbose = False)
    
    nonbinders = Y > censor_cutoff
    Y[nonbinders] = censor_cutoff
    model.fit(C,Y,nonbinders)
    print "--- AA interaction sparsity: %d / %d" % \
         ((np.abs(model.coef_) < 10**-6.0).sum(), len(model.coef_))
    return model.coef_

def split(data, start, stop):
    """
    Split a dataset for k-fold cross-validation
    """

    if len(data.shape) == 1:
        train = np.concatenate([data[:start], data[stop:]])
    else:
        train = np.vstack([data[:start], data[stop:]])
    test = data[start:stop]
    return train, test 

def evaluate_dataset(
        X_idx, Y, train_mask, allele, 
        initial_coef, n_iters, 
        binding_cutoff, censor_cutoff,
        log_targets = False, 
        regularization_weight_aa = 1, 
        regularization_weight_pos = 1):

    coeff_vec = initial_coef

    X_train_idx = X_idx[train_mask]
    X_test_idx = X_idx[~train_mask]
    assert X_train_idx.shape[1] == X_test_idx.shape[1]
    Y_train = Y[train_mask]
    Y_test = Y[~train_mask]
   

    print "Training baseline accuracy", \
        max(np.mean(Y_train <= binding_cutoff), 
            1 - np.mean(Y_train <= binding_cutoff))
    
    mask = Y_train > censor_cutoff
    Y_train_censored = Y_train.copy()
    Y_train_censored[mask] = censor_cutoff
    log_Y_train = np.log(Y_train)
    log_Y_train_censored = np.log(Y_train_censored)

    model = \
        CensoredLasso(
            regularization_weight = regularization_weight_pos, 
            verbose = False)
    
    for i in xrange(n_iters):
        print 
        print "- fitting regression model #%d (%s)" % ((i + 1), allele)
        
        assert len(coeff_vec)== (20*20)
        X_train = encode_inputs(X_train_idx, coeff_vec)

        X_test = encode_inputs(X_test_idx, coeff_vec)
        
        if i == 0:
            print "--- X_train shape", X_train.shape 
            print "--- X_test shape", X_test.shape
     
        print "--- coeff mean", np.mean(coeff_vec), "std", np.std(coeff_vec)
        
        last_iter = (i == n_iters - 1)
        
        if log_targets:
            model.fit(X_train, log_Y_train_censored, mask)
        else:
            model.fit(X_train, Y_train_censored, mask)

        pred = model.predict(X_test)
        if log_targets:
            pred = np.exp(pred)

        if not last_iter:
            model_weights = model.coef_
            print "--- Positional min weight abs:",\
                 np.abs(model_weights).min()
            print "--- Positional sparsity: %d / %d" % \
                (np.sum(np.abs(model_weights) < 10.0 ** -6), len(model_weights))
            coeff_vec = \
                estimate_pairwise_features(
                    X_train_idx, model_weights, Y_train, 
                    censor_cutoff = censor_cutoff, 
                    regularization_weight = regularization_weight_aa)
        
        pred_lte = pred <= binding_cutoff
        median_error = np.median(np.abs(pred-Y_test))
        actual_lte = Y_test <= binding_cutoff
        pred_gt = ~pred_lte 
        actual_gt = ~actual_lte 
        correct = (pred_lte & actual_lte) | (pred_gt & actual_gt)
        accuracy = np.mean(correct)

        sensitivity = np.mean(pred_lte[actual_lte])
        specificity = np.mean(actual_lte[pred_lte])
        auc = sklearn.metrics.roc_auc_score(actual_lte, 1.0 / pred)
        print "--- %d binders of %d identified" % ((correct & pred_lte).sum(), actual_lte.sum())
        print "--- median error", median_error 
        print "--- accuracy", accuracy 
        print "--- sensitivity", sensitivity 
        print "--- specificity", specificity
        print "--- AUC", auc 
    return accuracy, auc, sensitivity, specificity


def find_best_regularization_weights(
        X, Y, censor_cutoff, initial_coef, 
        candidate_weights, 
        binding_cutoff = 500, 
        n = 2000, 
        n_iters = 4):
    n = min(n, len(X) / 2)
    indices = np.arange(len(X))
    train_idx = indices[:n]
    test_idx = indices[n:2*n]
    X_train_indirect = X[train_idx]
    X_test_indirect = X[test_idx]
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    X_train_init = encode_inputs(X_train_indirect, initial_coef)
    C_train = Y_train > censor_cutoff
    Y_train[C_train] = censor_cutoff
    best_w_pos= None 
    best_w_aa = None
    best_auc = 0
    print "Testing for best regularization_weight: "
    for w_pos in candidate_weights:
        for w_aa in candidate_weights:

            pos_regression = CensoredLasso(regularization_weight = w_pos)
            aa_regression = CensoredLasso(regularization_weight = w_aa)
            X_train = X_train_init
            for i in xrange(n_iters):
                pos_regression.fit(X_train, Y_train, C_train)
                if i < n_iters - 1:
                    X_train_pairwise = \
                        encode_pairwise_coefficients(
                        X_train_indirect, 
                        pos_regression.coef_)


                    aa_regression.fit(X_train_pairwise, Y_train, C_train)
                    X_train = \
                        encode_inputs(
                            X_train_indirect, aa_regression.coef_)
            X_test = encode_inputs(X_train_indirect, aa_regression.coef_)
            pred = pos_regression.predict(X_test)
            print (Y_test <= binding_cutoff).mean(), (pred <= binding_cutoff).mean()
            
            auc = sklearn.metrics.roc_auc_score(
                Y_test <= binding_cutoff, -pred)
            print "aa = %f, pos = %f, auc = %f" % (w_aa, w_pos, auc)
            if auc > best_auc:
                best_auc = auc 
                best_w_aa = w_aa
                best_w_pos = w_pos 
        
    print "Best AUC %f w_aa %f w_pos" % (best_auc, w_aa, w_pos)
    return w_aa, w_pos 




def leave_one_out(X_idx, Y, W, alleles, 
                  binding_cutoff = 500, 
                  censor_cutoff = 2000, 
                  n_iters = 10,
                  log_targets = False, 
                  regularization_weights = \
                    [5, 1, 0.5, .1, .05, .01, .001, 0]):
    """
    X_idx : 2-dimensional array of integers with shape = (n_samples, n_features) 
        Elements are indirect references to elements of the feature encoding matrix
    
    Y : 1-dimensional array of floats with shape = (n_samples,)
        target IC50 values
    
    W : 1-dimensional array of floats with shape = (n_samples,)
        sample weights 
    
    alleles : str list 

    cutoff : int 
        Binding affinity threshold below which a peptide is considered a binder

    n_iters : int 
        Number of training iterations to refine feature matrix 
    """

    n_samples = len(Y)
    
    accuracies = []
    sensitivities = []
    specificities = []

    pmbec_coeff = pmbec.read_coefficients()
    pmbec_coeff_vec = feature_dictionary_to_vector(pmbec_coeff)

    
    unique_alleles = set(alleles)

    results = {}

    try:
        for allele in sorted(unique_alleles):
            print 
            print ">>>", allele 

            mask = ~np.array([x == allele for x in alleles])
            if (Y[mask] <= binding_cutoff).std() == 0 or \
                    (Y[~mask] <= binding_cutoff).std() == 0:
                print "Skipping %s" % allele
                continue 
            regularization_weight_aa, regularization_weight_pos = \
                find_best_regularization_weights(
                    X_idx[mask], Y[mask], 
                    censor_cutoff, pmbec_coeff_vec, 
                    regularization_weights) 
        
            accuracy, auc, sensitivity, specificity = \
                evaluate_dataset(
                    X_idx, Y, mask, allele, 
                    pmbec_coeff_vec, n_iters, 
                    binding_cutoff, censor_cutoff,
                    log_targets,
                    regularization_weight_aa, regularization_weight_pos)

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            accuracies.append(accuracy)
            results[allele] = (accuracy, auc)
    except KeyboardInterrupt:
        print "Results:"
        for allele in sorted(results.keys()):
            (pcc, auc) = results[allele]
            print "%s PCC %0.5f AUC %0.5f" % (allele, pcc, auc)

    print "Overall CV sensitivity =", np.mean(sensitivities)
    print "Overall CV specificity =", np.mean(specificities)
    print "Overall CV accuracy =", np.mean(accuracies)
        

