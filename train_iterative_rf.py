import numpy as np
import pandas as pd 
from collections import Counter 
import cPickle 

import pmbec 
from epitopes import amino_acid 
from parakeet import jit 
import scipy.weave 
import sklearn.linear_model
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
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

def encode_inputs(X_pair_indices, pairwise_feature_vec):
    """
    X_pair_indices : 2d array
        Indices of amino acid combinations for each (peptide, allele pseudosequence) entry
    pairwise_features : dict
        Maps from AA pair indices to continuous values 
    """ 

    code = """
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_dims; ++j) {
                X_encoded[i*n_dims+j] = \
                    pairwise_feature_vec[X_pair_indices[i*n_dims+j]];
            }
        }
    """
    X_pair_indices = np.ascontiguousarray(X_pair_indices)
    pairwise_feature_vec = np.ascontiguousarray(
        pairwise_feature_vec)
    n_samples, n_dims = X_pair_indices.shape 
    n_pairwise_features = len(pairwise_feature_vec)
    X_encoded = np.zeros((n_samples, n_dims), dtype=float)
    scipy.weave.inline(
        code, 
        arg_names=[
            "n_samples", 
            "n_dims", 
            "pairwise_feature_vec",
            "X_pair_indices",
            "X_encoded"
        ]
    )

    return X_encoded


def encode_pairwise_coefficients(X_idx, model_weights):
    n_samples, n_position_pairs = X_idx.shape 
    n_amino_acid_pairs = 20 * 20
    X_idx = np.ascontiguousarray(X_idx)
    model_weights = np.ascontiguousarray(model_weights)
    coeffs = np.zeros((n_samples, n_amino_acid_pairs), dtype=float)
    code = """
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_position_pairs; ++j) {
                int xi = X_idx[i * n_position_pairs + j];
                coeffs[i * n_amino_acid_pairs + xi] += model_weights[j];
            }
        }
    """
    scipy.weave.inline(
        code,
        arg_names=[
            "X_idx",
            "coeffs",
            "model_weights",
            "n_samples",
            "n_position_pairs",
            "n_amino_acid_pairs",
        ])
    return coeffs
   
def estimate_pairwise_features(X_idx, model_weights, Y_label):
   
    assert len(model_weights) == X_idx.shape[1], \
        "expected shape %s != %s" % (X_idx.shape[1], model_weights.shape)

    assert (X_idx.max() < 20 * 20), X_idx.max()
    C = encode_pairwise_coefficients(X_idx, model_weights)
    sgd_iters =  int(math.ceil(10.0 ** 6 / len(C)))
    model = sklearn.linear_model.SGDClassifier(
        shuffle = True, n_iter = sgd_iters, alpha = 0.01)
    model.fit(C, Y_label)
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

class CompoundClassifier(object):
    def __init__(
            self, 

            penalty = 'l2',
            loss = 'log',
            n_iter = 20,
            shuffle = True,
            alpha = 0.001):
        
        def mk_clf():
            return sklearn.ensemble.RandomForestClassifier(100)
            #return sklearn.linear_model.SGDClassifier(
            #    penalty = penalty, 
            #    loss = loss, 
            #    shuffle = shuffle, 
            #    n_iter = n_iter, 
            #    alpha = alpha)
        self.strong_binders = [mk_clf(), mk_clf()]
        self.weak_binders = [mk_clf(), mk_clf()]
        self.non_binders = [mk_clf(), mk_clf()]
        self.reweighting = [LogisticRegression(), LogisticRegression()]

    def _transform_to_probabilities(self, X, i = 0):
        P = np.zeros( (len(X), 3), dtype= float)    
        P[:, 0] = self.strong_binders[i].predict_proba(X)[:, 1]
        P[:, 1] = self.weak_binders[i].predict_proba(X)[:, 1]
        P[:, 2] =  self.non_binders[i].predict_proba(X)[:, 0]
        return P 

    def fit(self, X, Y):
        random_filter = np.random.randn(len(Y)) > 0
        for i in xrange(2):
            mask = random_filter == i
            X_in = X[mask]
            X_out = X[~mask]
            Y_in  = Y[mask]
            Y_out = Y[~mask]

            self.strong_binders[i].fit(X_in, Y_in<=150)
            self.weak_binders[i].fit(X_in, Y_in <= 500)
            self.non_binders[i].fit(X_in, Y_in>=5000)
            P = self._transform_to_probabilities(X_out, i)
            self.reweighting[i].fit(P, Y_out <= 500)

    def predict_proba(self, X):
        
        P = self._transform_to_probabilities(X)
        Y = np.zeros( (len(X),2), dtype=float)
        for clf in self.reweighting:
            Y += clf.predict_proba(P)
        Y /= len(self.reweighting)
        return Y 

def evaluate_dataset(
        X_idx, Y_IC50, Y_cat, train_mask, allele, 
        initial_coef, n_iters,
        binding_cutoff = 500):

    coeff_vec = initial_coef
    cat_mask = np.abs(Y_cat) > 0
    X_train_idx = X_idx[train_mask ]
    X_test_idx = X_idx[~train_mask]
    assert X_train_idx.shape[1] == X_test_idx.shape[1]
    Y_train = Y_IC50[train_mask] 
    train_lte = Y_train <= 500
    Y_test = Y_IC50[~train_mask] 
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
        alpha = 0.0005) 
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
            model = CompoundClassifier(n_iter = sgd_iters)
            model.fit(X_train, Y_train)
        else:
            model.fit(X_train, train_lte)
        
        pred = model.predict_proba(X_test)[:, 1]
        pred_lte = pred >= 0.5
        print "Predicted binders fraction", np.mean(pred_lte)
        
        auc = sklearn.metrics.roc_auc_score(actual_lte, pred)
        if not last_iter:
            model_weights = model.coef_.squeeze()
            print "--- Positional min weight abs:",\
                 np.abs(model_weights).min()
            print "--- Positional sparsity: %d / %d" % \
                (np.sum(np.abs(model_weights) < 10.0 ** -6), len(model_weights))
            coeff_vec = \
                estimate_pairwise_features(
                    X_train_idx, model_weights, train_lte)

        pred_gt = ~pred_lte 
        actual_gt = ~actual_lte
        correct = (pred_lte & actual_lte) | (pred_gt & actual_gt)
        accuracy = np.mean(correct)
        sensitivity = np.mean(pred_lte[actual_lte])
        specificity = np.mean(actual_lte[pred_lte])

        
        print "--- %d binders of %d identified" % (
            (correct & pred_lte).sum(), actual_lte.sum()
        )
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

def leave_one_out(X_idx, Y_IC50, Y_cat, alleles, 
                  n_iters = 5, 
                  binding_cutoff = 500,
                  output_file_name = "cv_results.csv"):
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

    pmbec_coeff = pmbec.read_coefficients()
    pmbec_coeff_vec = feature_dictionary_to_vector(pmbec_coeff)

    volume_ratio_dict = make_aa_volume_ratio_dictionary()
    volume_ratio_vec = feature_dictionary_to_vector(volume_ratio_dict)
    hydropathy_product_dict = make_aa_hydropathy_product_dictionary()
    hydropathy_product_vec = feature_dictionary_to_vector(hydropathy_product_dict)
    
    unique_human_alleles = set(a for a in alleles if a.startswith("HLA"))


    np.random.seed(1)
    results = {}
    output_file = open(output_file_name, 'w')
    output_file.write("Allele,PCC,AUC,Sensitivity,Specificity\n")
    try:
        for allele in sorted(unique_human_alleles):
            print 
            print ">>>", allele 

            mask = ~np.array([x == allele for x in alleles])
            if (Y_IC50[mask] <= binding_cutoff).std() == 0 or \
                    (Y_IC50[~mask] <= binding_cutoff).std() == 0:
                print "Skipping %s" % allele
                continue 
            
            accuracy, auc, sensitivity, specificity = \
                evaluate_dataset(
                    X_idx, Y_IC50, Y_cat, mask, allele, 
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


def load_training_data():
    X = np.load("X.npy")
    Y_IC50 = np.load("Y_IC50.npy")
    Y_cat = np.load("Y_category.npy")

    with open('alleles.txt', 'r') as f:
        alleles = [l.strip() for l in f.read().split("\n") if len(l) > 0]
    assert len(X) == len(Y_IC50)
    assert len(X) == len(Y_cat)
    assert len(X.shape) == 2
    assert len(alleles) == len(X)
    mask = \
        (Y_IC50 > 0) & ~np.isinf(Y_IC50) & ~np.isnan(Y_IC50) & (Y_IC50< 10**7)
    print "Drop %d entries with bad IC50 values" % (
        len(mask) - mask.sum()
    )
    X = X[mask]
    Y_IC50 = Y_IC50[mask]
    Y_cat = Y_cat[mask]
    alleles = [alleles[i] for i,b in enumerate(mask) if b]
    return X, Y_IC50, Y_cat, alleles

if __name__ == "__main__":
    X, Y_IC50, Y_cat, alleles = load_training_data()
    print "Loaded X.shape = %s" % (X.shape,)
    alleles, (X, Y_IC50, Y_cat) = shuffle(alleles, X, Y_IC50, Y_cat)
    leave_one_out(X, Y_IC50, Y_cat, alleles)