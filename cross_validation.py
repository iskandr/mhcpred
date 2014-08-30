
import numpy as np 
import pandas as pd 

import sklearn.linear_model
from parakeet import jit 



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

def estimate_pairwise_features(X_idx, model_weights, Y):
    Y = np.log(Y) #np.minimum(1.0, np.maximum(0.0, 1.0 - np.log(Y)/ np.log(50000)))

    assert len(model_weights) == X_idx.shape[1]
    assert (X_idx.max() < 20 * 20), X_idx.max()
    C = encode_pairwise_coefficients(X_idx, model_weights)
    model = sklearn.linear_model.Ridge()
    model.fit(C, Y)
    features = model.coef_ 
    return features 

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

def leave_one_out(X_idx, Y, W, alleles, cutoff = 500, n_iters = 25):
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

        print "Training baseline accuracy", max(np.mean(Y_train <= cutoff), 1 - np.mean(Y_train <= cutoff))
        
        model = LogLinearRegression()
        
            
           
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
            if last_iter:
                print "Training two-pass regression model"
                model = SelectiveRegressor()

            model.fit(X_train, Y_train, W_train)
            
            if last_iter:
                mask, pred = model.predict(X_test)
                print "Made predictions for: %d/%d" % (mask.sum(), len(mask))
                pred_lte = np.zeros(len(Y_test), dtype=bool)
                pred_lte[mask] = (pred <= cutoff)

                max_error = np.max(np.abs(pred-Y_test[mask]))
                median_error = np.median(np.abs(pred-Y_test[mask]))
            else:
                model_weights = model.coef_
                coeff_vec = estimate_pairwise_features(X_train_idx, model_weights, Y_train)
            
                pred = model.predict(X_test)
                
                pred_lte = pred <= cutoff
                max_error = np.max(np.abs(pred-Y_test))
                median_error = np.median(np.abs(pred-Y_test))
            
            actual_lte = Y_test <= cutoff
            pred_gt = ~pred_lte 
            actual_gt = ~actual_lte 
            correct = (pred_lte & actual_lte) | (pred_gt & actual_gt)
            accuracy = np.mean(correct)

            sensitivity = np.mean(pred_lte[actual_lte])
            specificity = np.mean(actual_lte[pred_lte])

            print " -- max error", max_error
            print "--- median error", median_error 
            print "--- accuracy", accuracy 
            print "--- sensitivity", sensitivity 
            print "--- specificity", specificity
            
        errors.append(median_error)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)

    print "Overall CV error  =", np.mean(errors)
    print "Overall CV sensitivity =", np.mean(sensitivities)
    print "Overall CV specificity =", np.mean(specificities)
    print "Overall CV accuracy =", np.mean(accuracies)
    return np.mean(errors)      
