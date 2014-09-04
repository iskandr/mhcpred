
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

def estimate_pairwise_features(X_idx, model_weights, Y, censor_cutoff = 2000):
    """
    Estimators:
    - ridge : Ridge Regression for all IC50 values

    - ...otherwise: minimize L1 norm s.t. IC50 <= 2000nM have exact same values, non-binders have predict IC50 > 2000nM

    cutoff : float 
        Binding affinity below which we want exact match of predicted values 
    """
    Y = np.log(Y) 

    assert len(model_weights) == X_idx.shape[1]
    assert (X_idx.max() < 20 * 20), X_idx.max()
    C = encode_pairwise_coefficients(X_idx, model_weights)
    
    model = CensoredLasso(regularization_weight = 0.001, verbose = False)
    log_cutoff = np.log(censor_cutoff)
    nonbinders = Y > log_cutoff
    Y[nonbinders] = log_cutoff
    model.fit(C,Y,nonbinders)
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

def leave_one_out(X_idx, Y, W, alleles, binding_cutoff = 500, censor_cutoff = 10000, n_iters = 15):
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
    
    errors = []
    accuracies = []
    sensitivities = []
    specificities = []
    pmbec_coeff = pmbec.read_coefficients()
    pmbec_coeff_vec = feature_dictionary_to_vector(pmbec_coeff)

    
    unique_alleles = set(alleles)

    for allele in sorted(unique_alleles):
        print ">>>", allele 

        coeff_vec = pmbec_coeff_vec

        allele_mask = [x == allele for x in alleles]
        mask = ~np.array(allele_mask)

        X_train_idx = X_idx[mask]
        X_test_idx = X_idx[~mask]
        assert X_train_idx.shape[1] == X_test_idx.shape[1]
        Y_train = Y[mask]
        Y_test = Y[~mask]
        W_train = W[mask]
        W_test = W[~mask]

        if (Y_train <= binding_cutoff).std() == 0 or (Y_test <= binding_cutoff).std() == 0:
            print "Skipping %s" % allele
            continue 
        print "Training baseline accuracy", \
            max(np.mean(Y_train <= binding_cutoff), 
                1 - np.mean(Y_train <= binding_cutoff))
        
        #model = LogLinearRegression()
        mask = Y_train >= censor_cutoff
        log_Y_train = np.log(Y_train)
        model = CensoredLasso(regularization_weight = 0.1, verbose = False)
        log_Y_train_censored = log_Y_train.copy()
        log_Y_train_censored[mask] = np.log(censor_cutoff)

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
            
            #if last_iter:
            #    model = SelectiveRegressor(cutoff = np.log(censor_cutoff))

            model.fit(X_train, log_Y_train_censored, mask)
            
            if last_iter:

                log_pred = model.predict(X_test)
                pred = np.exp(log_pred)
                mask = pred <= censor_cutoff
                print "Made predictions for: %d/%d" % (mask.sum(), len(mask))
                pred_lte = pred <= binding_cutoff
                max_error = np.max(np.abs(pred[mask]-Y_test[mask]))
                median_error = np.median(np.abs(pred[mask]-Y_test[mask]))
            else:
                model_weights = model.coef_
                print "Min weight abs:", np.abs(model_weights).min()
                print "Sparsity: %d / %d" % (np.sum(np.abs(model_weights) < 10.0 ** -5), len(model_weights))
                coeff_vec = estimate_pairwise_features(X_train_idx, model_weights, Y_train, censor_cutoff = censor_cutoff)
            
                pred = np.exp(model.predict(X_test))
                
                pred_lte = pred <= binding_cutoff
                max_error = np.max(np.abs(pred-Y_test))
                median_error = np.median(np.abs(pred-Y_test))
            
            actual_lte = Y_test <= binding_cutoff
            pred_gt = ~pred_lte 
            actual_gt = ~actual_lte 
            correct = (pred_lte & actual_lte) | (pred_gt & actual_gt)
            accuracy = np.mean(correct)

            sensitivity = np.mean(pred_lte[actual_lte])
            specificity = np.mean(actual_lte[pred_lte])
            auc = sklearn.metrics.roc_auc_score(actual_lte, -pred)
            print " -- max error", max_error
            print "--- median error", median_error 
            print "--- accuracy", accuracy 
            print "--- sensitivity", sensitivity 
            print "--- specificity", specificity
            print "--- AUC", auc 

        errors.append(median_error)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)

    print "Overall CV error  =", np.mean(errors)
    print "Overall CV sensitivity =", np.mean(sensitivities)
    print "Overall CV specificity =", np.mean(specificities)
    print "Overall CV accuracy =", np.mean(accuracies)
    return np.mean(errors)      


