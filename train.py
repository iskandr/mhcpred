import numpy as np
import pandas as pd 
from collections import Counter 
import cPickle 

import pmbec 
from epitopes import amino_acid 


from parakeet import jit 
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
def encode_pairwise_coefficients(
        X_idx, model_weights, censored_lasso = False):
    n_samples, n_position_pairs = X_idx.shape 
    n_amino_acid_pairs = 20 * 20
    coeffs = np.zeros((n_samples, n_amino_acid_pairs), dtype=float)
    for row_idx, x in enumerate(X_idx):
        for col_idx, xi  in enumerate(x):
            coeffs[row_idx, xi] += model_weights[col_idx]
    return coeffs

def estimate_pairwise_features(X_idx, model_weights, Y):
   
    assert len(model_weights) == X_idx.shape[1], \
        "expected shape %s != %s" % (X_idx.shape[1], model_weights.shape)

    assert (X_idx.max() < 20 * 20), X_idx.max()
    C = encode_pairwise_coefficients(X_idx, model_weights)
    sgd_iters =  int(math.ceil(10.0 ** 6 / len(C)))
    model = model = sklearn.linear_model.SGDClassifier(shuffle = True, n_iter = sgd_iters, alpha = 0.005)
    #model = sklearn.linear_model.LogisticRegression()
    model.fit(C, Y <= 500)
    return model.coef_.squeeze()

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
        binding_cutoff = 500):

    coeff_vec = initial_coef

    X_train_idx = X_idx[train_mask]
    X_test_idx = X_idx[~train_mask]
    assert X_train_idx.shape[1] == X_test_idx.shape[1]
    Y_train = Y[train_mask] 
    train_lte = Y_train <= 500
    Y_test = Y[~train_mask] 
    actual_lte = Y_test <= 500
   

    print "Training baseline accuracy", \
        max(np.mean(actual_lte), 
            1 - np.mean(actual_lte))
    
    
    sgd_iters =  int(math.ceil(10.0 ** 6 / len(X_train_idx)))
    model = sklearn.linear_model.SGDClassifier(
        penalty = 'l1', 
        loss = 'log', 
        shuffle = True, 
        n_iter = sgd_iters, 
        alpha = 0.0001) 
    #model = sklearn.linear_model.LogisticRegression()
    for i in xrange(n_iters):
        print 
        print "- fitting regression model #%d (%s)" % ((i + 1), allele)
        
        assert len(coeff_vec)== (20*20), "Expected 400, got %s" % (len(coeff_vec),)
        X_train = encode_inputs(X_train_idx, coeff_vec)

        X_test = encode_inputs(X_test_idx, coeff_vec)
        
        if i == 0:
            print "--- X_train shape", X_train.shape 
            print "--- X_test shape", X_test.shape
     
        print "--- coeff mean", np.mean(coeff_vec), "std", np.std(coeff_vec)
        
        last_iter = (i == n_iters - 1)
        
        if last_iter:
            #model = sklearn.ensemble.RandomForestClassifier(300)
            #model = sklearn.ensemble.RandomForestRegressor(300)
            #model = sklearn.linear_model.LinearRegression()
            model = LassoCV(normalize = True)
            #model.fit(X_train, np.log(Y_train))
            
            #model = CensoredLasso(regularization_weight = 0.0001)
            censor_cutoff = 20000
            C = Y_train >= censor_cutoff
            #Y_train[C] = censor_cutoff
            model.fit(X_train, np.log(Y_train))
            pred = np.exp(model.predict(X_test))
            print "--- Pred", pred[:10].astype(int)
            print "--- True", Y_test[:10].astype(int)
            pred_lte = (pred <= binding_cutoff)

        else:
            model.fit(X_train, train_lte)
            pred = model.predict_proba(X_test)[:, 1]
            
            pred_lte = pred >= 0.5
        print "Predicted binders fraction", np.mean(pred_lte)
        
        if  last_iter:
            auc = sklearn.metrics.roc_auc_score(actual_lte, -pred)
        else:
            auc = sklearn.metrics.roc_auc_score(actual_lte, pred)
            model_weights = model.coef_.squeeze()
            print "--- Positional min weight abs:",\
                 np.abs(model_weights).min()
            print "--- Positional sparsity: %d / %d" % \
                (np.sum(np.abs(model_weights) < 10.0 ** -6), len(model_weights))
            coeff_vec = \
                estimate_pairwise_features(
                    X_train_idx, model_weights, Y_train)

        pred_gt = ~pred_lte 
        actual_gt = ~actual_lte
        correct = (pred_lte & actual_lte) | (pred_gt & actual_gt)
        accuracy = np.mean(correct)
        sensitivity = np.mean(pred_lte[actual_lte])
        specificity = np.mean(actual_lte[pred_lte])

        
        print "--- %d binders of %d identified" % ((correct & pred_lte).sum(), actual_lte.sum())
        print "--- accuracy", accuracy 
        print "--- sensitivity", sensitivity 
        print "--- specificity", specificity
        print "--- AUC", auc 
    return accuracy, auc, sensitivity, specificity


def make_aa_volume_ratio_dictionary():
    result = {}
    volumes = epitopes.amino_acid.volume.value_dict
    for a in AMINO_ACID_LETTERS:
        va = volumes[a]
        for b in AMINO_ACID_LETTERS:
            vb = volumes[b]
            result[a + b] = np.log(va/vb)
    return result

def make_aa_hydropathy_product_dictionary():
    result  = {}
    d = epitopes.amino_acid.hydropathy.value_dict
    for a in AMINO_ACID_LETTERS:
        ha = d[a]
        for b in AMINO_ACID_LETTERS:
            hb = d[b]
            result[a + b] = ha * hb 
    return result

def leave_one_out(X_idx, Y, alleles, 
                  n_iters = 10, 
                  binding_cutoff = 500,
                  output_file_name = "cv_results.csv"):
    """
    X_idx : 2-dimensional array of integers with shape = (n_samples, n_features) 
        Elements are indirect references to elements of the feature encoding matrix
    
    Y : 1-dimensional array of floats with shape = (n_samples,)
        target IC50 values

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
    aucs = []

    pmbec_coeff = pmbec.read_coefficients()
    pmbec_coeff_vec = feature_dictionary_to_vector(pmbec_coeff)

    volume_ratio_dict = make_aa_volume_ratio_dictionary()
    volume_ratio_vec = feature_dictionary_to_vector(volume_ratio_dict)
    hydropathy_product_dict = make_aa_hydropathy_product_dictionary()
    hydropathy_product_vec = feature_dictionary_to_vector(hydropathy_product_dict)
    
    unique_human_alleles = set(a for a in alleles if a.startswith("HLA"))

    results = {}
    np.random.seed(1)
    index = np.arange(len(Y))
    

    rand_vec = np.random.randn(len(hydropathy_product_vec))
    output_file = open(output_file_name, 'w')
    output_file.write("Allele,PCC,AUC,Sensitivity,Specificity\n")
    try:
        for allele in sorted(unique_human_alleles):
            print 
            print ">>>", allele 

            mask = ~np.array([x == allele for x in alleles])
            if (Y[mask] <= binding_cutoff).std() == 0 or \
                    (Y[~mask] <= binding_cutoff).std() == 0:
                print "Skipping %s" % allele
                continue 
            

            accuracy, auc, sensitivity, specificity = \
                evaluate_dataset(
                    X_idx, Y, mask, allele, 
                    pmbec_coeff_vec, n_iters)
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
        (pcc, auc) = results[allele]
        print "%s PCC %0.5f AUC %0.5f" % (allele, pcc, auc)
    print "Overall CV sensitivity =", np.mean(sensitivities)
    print "Overall CV specificity =", np.mean(specificities)
    print "Overall CV accuracy =", np.mean(accuracies)
        



def shuffle(X, Y, alleles):
    n = len(Y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    alleles = [alleles[i] for i in indices]
    return X, Y, alleles


def load_training_data():
    print "Loading X"
    X = np.load("X.npy")
    print "Loading Y"
    Y = np.load("Y_IC50.npy")

    with open('alleles.txt', 'r') as f:
        alleles = [l.strip() for l in f.read().split("\n") if len(l) > 0]
    assert len(X) == len(Y)
    assert len(X.shape) == 2
    assert len(alleles) == len(X)
    mask = (Y > 0) & ~np.isinf(Y) & ~np.isnan(Y) & (Y< 10**7)
    X = X[mask]
    Y = Y[mask]
    alleles = [alleles[i] for i,b in enumerate(mask) if b]
    return X, Y, alleles

if __name__ == "__main__":
    X, Y, alleles = load_training_data()
    print "Loaded X.shape = %s" % (X.shape,)
    X, Y, alleles = shuffle(X, Y, alleles)
    leave_one_out(X,Y,alleles)