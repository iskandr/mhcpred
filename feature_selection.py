import argparse
import collections

import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser(
    description=
        """
        Out-of-core feature selection from an HDF file
        """
)

parser.add_argument(
    "--input-file",
    default="pairwise_features.hdf",
    help="Input HDF5 file"
)

parser.add_argument(
    "--iters",
    type = int,
    default = 100,
    help="How many subset models to train"
)

parser.add_argument(
    "--feature-fraction",
    type = float,
    default = 0.05,
    help="What portion of features to use in each model"
)

parser.add_argument(
    "--sample-fraction",
    type = float,
    default = 0.01,
    help="What portion of sample to train each model on"
)

parser.add_argument(
    "--ic50-cutoff",
    type = float,
    default = 5000,
    help="Threshold to separate binders from non-binders"
)


def find_best_threshold_accuracy(x, y):
    thresholds = np.unique(x)
    best_accuracy = 0
    best_threshold = None
    for t in thresholds:
        mask = (x<=t)
        acc_below = np.mean(mask == y)
        acc_above = np.mean(~mask == y)
        accuracy = max(acc_below, acc_above)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
    return best_accuracy, best_threshold

def find_threshold_pairs(x1, x2, y):
    thresholds1 = np.unique(x1)
    thresholds2 = np.unique(x2)
    best_accuracy = 0
    best_threshold = None
    for t1 in thresholds1:
        mask1 = (x1<=t1)
        for t2 in thresholds2:
            mask2 = (x2<=t2)
            accuracy = max(
                np.mean(mask == y)
                for mask in
                [
                  mask1 & mask2,
                  mask1 & ~mask2,
                  ~mask1 & mask2,
                  ~mask1 & ~mask2
                ]
            )
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = (t1, t2)
    return best_accuracy, best_threshold


if __name__ == "__main__":
    args = parser.parse_args()
    print "ARGUMENTS"
    print args

    assert args.sample_fraction > 0
    assert args.sample_fraction <= 0.5, \
        "Can't use more than 50% of the samples for training iterations, " \
        "not enough left for testing"
    assert 0 < args.feature_fraction <= 1.0
    assert args.iters > 0

    f = h5py.File(args.input_file)
    feature_names = [
        name
        for name in f.iterkeys()
        if not name.upper().startswith("Y")
    ]
    print "Found %d features in %s" % (len(feature_names), args.input_file)
    y = f['Y'][:] <= args.ic50_cutoff
    n_samples = len(y)
    n_features = len(feature_names)
    n_samples_per_iter = int(n_samples * args.sample_fraction)
    n_features_per_iter = int(n_features * args.feature_fraction)
    # these will get shuffled to create subsets
    all_sample_indices = np.arange(n_samples)
    all_feature_indices = np.arange(n_features)
    print "Samples per iter: %d / %d" % (n_samples_per_iter, n_samples)
    print "Features per iter: %d / %d" % (n_features_per_iter, n_features)

    feature_counts = collections.Counter()
    feature_nonzero = collections.Counter()

    # for each split try different hyperparameters and look
    # at non-zero coefficients in the model with best
    # predictive accuracy
    models = [
        LogisticRegression(penalty='l1', C = 1),
        LogisticRegression(penalty='l1', C = 0.1),
        LogisticRegression(penalty='l1', C = 0.01),
        LogisticRegression(penalty='l1', C = 0.001),
    ]

    for i in xrange(args.iters):
        np.random.shuffle(all_feature_indices)
        np.random.shuffle(all_sample_indices)
        feature_indices = all_feature_indices[:n_features_per_iter]
        training_indices = all_sample_indices[:n_samples_per_iter]
        testing_indices = \
            all_sample_indices[n_samples_per_iter:2*n_samples_per_iter]
        Y_train = y[training_indices]
        Y_test = y[testing_indices]

        my = Y_test.mean()
        baseline = max(my, 1.0 - my)
        print "Baseline accuracy for iter #%d: %0.4f" % (i+1, baseline)
        X_train = []
        X_test = []
        for feature_idx in feature_indices:
            name = feature_names[feature_idx]
            col = f[name][:]
            train = col[training_indices]
            test = col[testing_indices]
            X_train.append(train)
            X_test.append(test)
        X_train = np.array(X_train).T
        X_test = np.array(X_test).T
        best_model = None
        best_accuracy = 0

        print "-- train shape = %s, test shape = %s)" % (
            X_train.shape,
            X_test.shape,
        )
        for model in models:

            model.fit(X_train, Y_train)
            pred = model.predict(X_test)
            accuracy = np.mean(pred == Y_test)
            print " * ", model, accuracy
            if accuracy > best_accuracy:
                best_model = model
                best_accuracy = accuracy
        cutoff = 0.000001
        coeff = best_model.coef_.ravel()
        abs_coeff = np.abs(coeff)
        print "-- Fraction nonzero coeffs: ", \
            np.mean(abs_coeff > cutoff)
        if best_accuracy < baseline:
            print "Skipping iteration #%d due to low accuracy" % (i+1)
            continue
        # value of a predictor is how much better than baseline it did
        value = best_accuracy - baseline
        print "-- Improvement over baseline: %0.4f" % value
        total = abs_coeff.sum()
        fractions = abs_coeff/ total
        for i, p in enumerate(fractions):
            feature_idx = feature_indices[i]
            name = feature_names[feature_idx]
            feature_counts[name] += 1
            feature_nonzero[name] += value * p

    feature_scores = collections.Counter()
    for name, v in feature_nonzero.iteritems():
        feature_scores[name] = v / float(feature_counts[name])

    for name, acc in feature_scores.most_common()[::-1]:
        print name, acc, "(%d)" % feature_counts[name]
