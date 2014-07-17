import numpy as np
import pandas as pd 
from collections import Counter 

import pmbec 
from epitopes import amino_acid 

from parakeet import jit 
import sklearn.linear_model
import sklearn.svm 
import sklearn.ensemble
import sklearn.decomposition 

AMINO_ACID_LETTERS =list(sorted([
	'G', 'P',
	'A', 'V',
	'L', 'I',
	'M', 'C',
	'F', 'Y', 
	'W', 'H', 
	'K', 'R',
	'Q', 'N', 
	'E', 'D',
	'S', 'T',
]))

AMINO_ACID_PAIRS = ["%s%s" % (x,y) for y in AMINO_ACID_LETTERS for x in AMINO_ACID_LETTERS]

AMINO_ACID_PAIR_POSITIONS = dict( (y, x) for x, y in enumerate(AMINO_ACID_PAIRS))

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
	print "--- Fitting model for pairwise features w/ C.shape = %s, C.mean = %0.4f C.std = %0.4f" % \
		(C.shape, C.mean(), C.std())
	model = sklearn.linear_model.Ridge()
	model.fit(C, Y)
	features = model.coef_ 
	return features 

def generate_training_data(binding_data_filename = "mhc1.csv", mhc_seq_filename = "MHC_aa_seqs.csv"):
	df_peptides = pd.read_csv(binding_data_filename).reset_index()

	print "Loaded %d peptides" % len(df_peptides)

	df_peptides['MHC Allele'] = df_peptides['MHC Allele'].str.replace('*', '').str.strip()
	df_peptides['Epitope']  = df_peptides['Epitope'].str.strip().str.upper()

	print "Peptide lengths"
	print df_peptides['Epitope'].str.len().value_counts()


	mask = df_peptides['Epitope'].str.len() == 9
	mask &= df_peptides['IC50'] <= 10**6
	df_peptides = df_peptides[mask]
	print "Keeping %d peptides (length >= 9)" % len(df_peptides)


	groups = df_peptides.groupby(['MHC Allele', 'Epitope'])
	grouped_ic50 = groups['IC50']
	grouped_std = grouped_ic50.std() 
	grouped_count = grouped_ic50.count() 
	duplicate_std = grouped_std[grouped_count > 1]
	duplicate_count = grouped_count[grouped_count > 1]
	print "Found %d duplicate entries in %d groups" % (duplicate_count.sum(), len(duplicate_count))
	print "Std in each group: %0.4f mean, %0.4f median" % (duplicate_std.mean(), duplicate_std.median())
	df_peptides = grouped_ic50.median().reset_index()

	# reformat HLA allales 'HLA-A*03:01' into 'HLA-A0301'
	peptide_alleles = df_peptides['MHC Allele']
	peptide_seqs = df_peptides['Epitope']
	peptide_ic50 = df_peptides['IC50']
	
	print "%d unique peptide alleles" % len(peptide_alleles.unique())
	
	df_mhc = pd.read_csv(mhc_seq_filename)
	print "Loaded %d MHC alleles" % len(df_mhc)


	mhc_alleles = df_mhc['Allele'].str.replace('*', '')
	mhc_seqs = df_mhc['Residues']

	assert len(mhc_alleles) == len(df_mhc)
	assert len(mhc_seqs) == len(df_mhc)

	print list(sorted(peptide_alleles.unique()))
	print mhc_alleles[:20]
	print "%d common alleles" % len(set(mhc_alleles).intersection(set(peptide_alleles.unique())))
	print "Missing allele sequences for %s" % set(peptide_alleles.unique()).difference(set(mhc_alleles))

	mhc_seqs_dict = {}
	for allele, seq in zip(mhc_alleles, mhc_seqs):
		mhc_seqs_dict[allele] = seq 


	X = []
	W = []
	Y = []
	n_dims = 9 * len(mhc_seqs[0])
	for peptide_idx, allele in enumerate(peptide_alleles):
		if allele in mhc_seqs_dict:
			allele_seq = mhc_seqs_dict[allele]
			peptide = peptide_seqs[peptide_idx]
			n_peptide_letters = len(peptide)
			n_mhc_letters = len(allele_seq)
			ic50 = peptide_ic50[peptide_idx]
			print peptide_idx, allele, peptide, allele_seq, ic50
			for start_pos in xrange(0, n_peptide_letters - 8):
				stop_pos = start_pos + 9
				peptide_substring = peptide[start_pos:stop_pos]
				vec = [AMINO_ACID_PAIR_POSITIONS[peptide_letter + mhc_letter] 
				       for peptide_letter in peptide_substring
				       for mhc_letter in allele_seq]

				"""
				# add interaction terms for neighboring residues on the peptide
				for i, peptide_letter in enumerate(peptide_substring):
					if i > 0:
						before = peptide_substring[i - 1]
						vec.append(AMINO_ACID_PAIR_POSITIONS[before + peptide_letter])
					if i < 8:
						after = peptide_substring[i + 1]
						vec.append(AMINO_ACID_PAIR_POSITIONS[peptide_letter + after] )
				"""
				X.append(np.array(vec))
				Y.append(ic50)
				weight = 1.0 / (n_peptide_letters - 8)
				W.append(weight)
	X = np.array(X)
	W = np.array(W)
	Y = np.array(Y)

	print "Generated data shape", X.shape
	assert len(W) == X.shape[0]
	assert len(Y) == X.shape[0]
	return X, W, Y



class LogLinearRegression(sklearn.linear_model.Ridge):
	def fit(self, X, Y, sample_weight = None):
		self._max_value = np.max(Y)
		Y = np.log(Y)
		return sklearn.linear_model.Ridge.fit(self, X, Y)


	def predict(self, X):
		transformed_Y = sklearn.linear_model.Ridge.predict(self, X)
		raw_values = np.exp(transformed_Y)
		return np.minimum(self._max_value, raw_values)

class TwoPassRegressor(object):


	def fit(self,X,Y,W=None):
		category_base = 100
		categories =  np.maximum(0, (np.log10(Y) / np.log10(category_base)).astype('int')) 
		self.first_pass = sklearn.ensemble.RandomForestClassifier(n_estimators = 20) #sklearn.linear_model.LogisticRegression()
		self.first_pass.fit(X, categories)
		
		Y = np.log(Y)
		self.regressors = [None] * (np.max(categories) + 1)
		for category in np.unique(categories):
			mask = categories == category
			print "-- Category #%d (base %d): %d samples" % (category, category_base, mask.sum())
			regressor = sklearn.linear_model.RidgeCV()
			regressor.fit(X[mask], Y[mask])
			self.regressors[category] = regressor
		return self


	def predict(self, X):
		probs = self.first_pass.predict_proba(X)
		combined = np.zeros(X.shape[0], dtype=float)
		weights = np.zeros_like(combined)

		for category_idx in xrange(probs.shape[1]):
			pred = self.regressors[category_idx].predict(X)
			prob = probs[:, category_idx] 
			combined += prob * pred 
		return np.exp(combined) 


def split(data, start, stop):
	if len(data.shape) == 1:
		train = np.concatenate([data[:start], data[stop:]])
	else:
		train = np.vstack([data[:start], data[stop:]])
	test = data[start:stop]
	return train, test 

def shuffle(X, Y, W):
	n = len(Y)
	indices = np.arange(n)
	np.random.shuffle(indices)
	X = X[indices]
	Y = Y[indices]
	W = W[indices]
	return X, Y, W

def load_training_data():
	print "Loading X"
	X = np.load("X.npy")
	print "Loading Y"
	Y = np.load("Y.npy")
	print "Loading W"
	W = np.load("W.npy")
	assert len(X) == len(Y)
	assert len(W) == len(Y)
	assert len(X.shape) == 2
	return X, Y, W

def save_training_data(X, Y, W):
	print "Saving to disk..."
	np.save("X.npy", X)
	np.save("W.npy", W)
	np.save("Y.npy", Y)

def cross_validation(X_idx, Y, W, n_splits = 5):
	"""
	X_idx : 2-dimensional array of integers with shape = (n_samples, n_features) 
		Elements are indirect references to elements of the feature encoding matrix
	
	Y : 1-dimensional array of floats with shape = (n_samples,)
		target IC50 values
	
	W : 1-dimensional array of floats with shape = (n_samples,)
		sample weights 
	
	n_splits : int
	"""

	n_samples = len(Y)
	split_size = n_samples / n_splits
	
	errors = []
	accuracies = []
	sensitivities = []
	specificities = []
	pmbec_coeff = pmbec.read_coefficients()
	pmbec_coeff_vec = feature_dictionary_to_vector(pmbec_coeff)

	
	for split_idx in xrange(n_splits):
		coeff_vec = pmbec_coeff_vec
		test_start = split_idx * split_size
		test_stop = min((split_idx + 1) * split_size, n_samples)
		
		print "Split #%d" % (split_idx+1), "n =",  n_samples - (test_stop - test_start)
		X_train_idx, X_test_idx = split(X_idx, test_start, test_stop)
		assert len(X_train_idx.shape) == len(X_test_idx.shape)
		assert X_train_idx.shape[1] == X_test_idx.shape[1]
		Y_train, Y_test = split(Y, test_start, test_stop)
		W_train, W_test = split(W, test_start, test_stop)
		print "Training baseline accuracy", max(np.mean(Y_train <= 500), 1 - np.mean(Y_train <= 500))
		
		model = LogLinearRegression()
		
		n_iters = 10
		for i in xrange(n_iters):
			print 
			print "- fitting regression model #%d" % (i + 1)
			
			assert len(coeff_vec)== (20*20)
			X_train = encode_inputs(X_train_idx, coeff_vec)
			X_test = encode_inputs(X_test_idx, coeff_vec)
			
			print "--- X_train shape", X_train.shape 
			print "--- X_test shape", X_test.shape
			
			print "--- coeff first ten entries:", coeff_vec[0:10]
			print "--- coeff mean", np.mean(coeff_vec), "std", np.std(coeff_vec)
			
			last_iter = (i == n_iters - 1)
			if last_iter:
				print "Training two-pass regression model"
				model = TwoPassRegressor()

			model.fit(X_train, Y_train, W_train)
			
			if not last_iter:
				model_weights = model.coef_
				coeff_vec = estimate_pairwise_features(X_train_idx, model_weights, Y_train)
			
			pred = model.predict(X_test)
			
			pred_lte = pred <= 500
			actual_lte = Y_test <= 500
			pred_gt = ~pred_lte 
			actual_gt = ~actual_lte 
			correct = (pred_lte & actual_lte) | (pred_gt & actual_gt)
			total_weights = np.sum(W_test)
			accuracy = np.sum(W_test * correct) / total_weights

			sensitivity = np.sum(W_test[actual_lte] * correct[actual_lte]) / np.sum(W_test[actual_lte])
			specificity = np.sum(W_test[pred_lte] * correct[pred_lte]) / np.sum(W_test[pred_lte])

			split_error = np.sum(np.abs(pred-Y_test) * W_test) / np.sum(W_test)
						
			print "--- error:", split_error

			print " -- max error", np.max(np.abs(pred-Y_test))
			print "--- mean error", np.mean(np.abs(pred-Y_test))
			
			print "--- median error", np.median(np.abs(pred-Y_test))
			print "--- accuracy", accuracy 
			print "--- sensitivity", sensitivity 
			print "--- specificity", specificity
			
		errors.append(np.median(np.abs(pred-Y_test)))
		sensitivities.append(sensitivity)
		specificities.append(specificity)
		accuracies.append(accuracy)

	print "Overall CV error  =", np.mean(errors)
	print "Overall CV sensitivity =", np.mean(sensitivities)
	print "Overall CV specificity =", np.mean(specificities)
	print "Overall CV accuracy =", np.mean(accuracies)
	return np.mean(errors)		



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
		X,W,Y = generate_training_data()
		save_training_data(X, Y, W)

	if args.fit:
		if "X" not in locals() or "Y" not in locals() or "W" not in locals():
			X, Y, W = load_training_data()

		X, Y, W = shuffle(X, Y, W)

		cross_validation(X,Y,W)