import numpy as np
import pandas as pd 
from collections import Counter 
import cPickle 

import pmbec 
from epitopes import amino_acid 


from parakeet import jit 
import sklearn.linear_model
import sklearn.svm 
import sklearn.ensemble
import sklearn.decomposition 

from log_linear_regression import LogLinearRegression
from two_pass_regressor import TwoPassRegressor
from selective_regressor import SelectiveRegressor 
from generate_training_data import generate_training_data
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS
from cross_validation import leave_one_out




def load_training_data():
	print "Loading X"
	X = np.load("X.npy")
	print "Loading Y"
	Y = np.load("Y.npy")
	print "Loading W"
	W = np.load("W.npy")
	
	with open('alleles.txt', 'r') as f:
		alleles = [l.strip() for l in f.read().split("\n") if len(l) > 0]
	
	assert len(X) == len(Y)
	assert len(W) == len(Y)
	assert len(X.shape) == 2
	assert len(alleles) == len(X)

	return X, Y, W, alleles

def save_training_data(X, Y, W, alleles):
	print "Saving to disk..."
	np.save("X.npy", X)
	np.save("W.npy", W)
	np.save("Y.npy", Y)
	with open('alleles.txt', 'w') as f:
		for allele in alleles:
			f.write(allele)
			f.write("\n")


def shuffle(X, Y, W, alleles):
	n = len(Y)
	indices = np.arange(n)
	np.random.shuffle(indices)
	X = X[indices]
	Y = Y[indices]
	W = W[indices]
	alleles = [alleles[i] for i in indices]
	return X, Y, W, alleles



if __name__ == '__main__':
	
	import argparse
	parser = argparse.ArgumentParser(description='Generate training data for MHC binding prediction and use it to train regressors')
	parser.add_argument('--generate',  action='store_true', default=False)
	parser.add_argument('--fit', action='store_true', default=False)

	args = parser.parse_args()
	print "Commandline Args:"
	print args
	print 

	if args.generate:
		X,W,Y,alleles = generate_training_data()
		save_training_data(X, Y, W, alleles)

	if args.fit:
		if "X" not in locals() or "Y" not in locals() or "W" not in locals():
			X, Y, W, alleles = load_training_data()

		X, Y, W, alleles = shuffle(X, Y, W, alleles)

		leave_one_out(X,Y,W, alleles)