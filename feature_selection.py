import argparse
import collections

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(
    description=
        """
        Out-of-core feature selection from an HDF file
        """
)

parser.add_argument(
    "--input-file",
    default="pairwise_features.hdf",
    help="Input HDF5 file",
    required=True,
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
    "--target",
    required=True,
    help="Name of target column"
)

parser.add_argument(
    "--ignore",
    default="",
    help="Comma separated list of column names to ignore"
)

parser.add_argument(
    "--sample-attributes",
    default="",
    help="Comma separated list of columns to keep associated with each sample",
)

parser.add_argument(
    "--target-threshold",
    type = float,
    help="Threshold to turn continuous target into label categories"
)

parser.add_argument(
    "--model-fit-intercept",
    type = bool,
    default = True,
    help="Should trained models augment data with an intercept column?"
)

parser.add_argument(
    "--keep-feature-fraction",
    type=float,
    default=0.1,
    help="What fraction of features to keep"
)

parser.add_argument(
    "--output-data-file",
    type=str,
    default="selected.npz",
    help="Where to write selected features as an npz file"
)

parser.add_argument(
    "--use-pytables",
    help="Use PyTables instead of h5py to read HDF5 file",
    default=False,
    action="store_true",
)

parser.add_argument(
    "--balance-classes",
    help="Assign equal weight to pos/neg errors (for unbalanced data)",
    default=False,
    action="store_true"
)

parser.add_argument(
    "--feature-importance-cutoff",
    default=0.000001,
    type=float,
    help="How small can a feature weight/importance get before we ignore it?"
)

"""
Imitate the behavior of h5py for data generated by PyTables which
can't be loaded with h5py
"""
class PyTablesDataset(object):
    def __init__(self, field):
        self.field = field

    def __getitem__(self, arg):
        return self.field.read()[arg]

class PyTablesFile(object):
    def __init__(self, filename):
        import tables
        self.t = tables.open_file(args.input_file)

    def __getitem__(self, name):
        return PyTablesDataset(getattr(self.t.root, name))

    def __contains__(self, name):
        return hasattr(self.t.root, name)

    def keys(self):
        return [subnode.name for subnode in self.t.getNode("/")]

    def iterkeys(self):
        return iter(self.keys())

if __name__ == "__main__":
    args = parser.parse_args()

    if args.use_pytables:
        import tables
        f = PyTablesFile(args.input_file)
    else:
        import h5py
        f = h5py.File(args.input_file)


    print "ARGUMENTS"
    print args

    assert args.sample_fraction > 0
    assert args.sample_fraction <= 0.5, \
        "Can't use more than 50% of the samples for training iterations, " \
        "not enough left for testing"
    assert 0 < args.feature_fraction <= 1.0
    assert args.iters > 0

    sample_attribute_names = [x for x in args.sample_attributes.split(",") if x]
    for attr_name in sample_attribute_names:
        assert attr_name in f, \
        "Attribute '%s' not found in %s" % (attr_name, args.input_file)

    target = args.target
    ignore_columns = [x for x in args.ignore.split(",") if x]
    ignore_columns += sample_attribute_names
    ignore_columns += [target]

    assert target in f, \
        "Target column '%s' not found in %s" % (target, args.input_file)
    y = f[target][:]

    if args.target_threshold:
        y = y <= args.target_threshold
    else:
        unique_vals = np.unique(y)
        assert len(unique_vals) == 2, \
            "Expected binary label, use --target-threshold for float targets"

    feature_names = [
        name
        for name in f.iterkeys()
        if not name in ignore_columns
    ]

    n_samples = len(y)
    n_features = len(feature_names)
    n_samples_per_iter = int(n_samples * args.sample_fraction)
    n_features_per_iter = int(n_features * args.feature_fraction)

    bad_cols = set([])
    # checking features for NaN and infinite
    for i, feature_name in enumerate(sorted(feature_names)):
        print "Feature %d/%d: %s" % (i, n_features, feature_name)
        x = f[feature_name][:]

        n_nan = np.isnan(x).sum()
        if n_nan > 0:
            print "-- # NaN: %d" % n_nan
            bad_cols.add(feature_name)

        n_inf = np.isinf(x).sum()
        if n_inf > 0:
            print "-- # inf: %d" % n_inf
            bad_cols.add(feature_name)

    if len(bad_cols) > 0:
        feature_names = [x for x in feature_names if x not in bad_cols]
        n_features = len(feature_names)
        print "Bad columns: %s" % bad_cols
    print "Samples per iter: %d / %d" % (n_samples_per_iter, n_samples)
    print "Features per iter: %d / %d" % (n_features_per_iter, n_features)

     # these will get shuffled to create subsets
    all_sample_indices = np.arange(n_samples)
    all_feature_indices = np.arange(n_features)

    feature_counts = collections.Counter()
    feature_nonzero = collections.Counter()

    # for each split try different hyperparameters and look
    # at non-zero coefficients in the model with best
    # predictive accuracy
    intercept = args.model_fit_intercept

    class_weight = 'auto' if args.balance_classes else None
    lr_models = [
        LogisticRegression(
            penalty='l1',
            C = c,
            fit_intercept=intercept,
            class_weight=class_weight)
        for c in [10, 1, 0.1, 0.01, 0.001]
    ]

    rf_models = [
        RandomForestClassifier(n_estimators=15, criterion=c)
        for c in ["gini", "entropy"]
    ]

    models = lr_models + rf_models

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
        baseline_acc = max(my, 1.0 - my)
        baseline_auc = 0.5
        print "Baseline accuracy for iter #%d: %0.4f" % (i+1, baseline_acc)
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
        X_mean = X_train.mean(axis=0)
        X_train -= X_mean
        X_test -= X_mean


        best_model = None
        best_accuracy = 0
        best_auc = 0

        print "-- train shape = %s, test shape = %s)" % (
            X_train.shape,
            X_test.shape,
        )
        for model in models:
            # for models that don't take a class balancing parameter
            # we have to manually reweight the samples by their inverse class
            # frequency
            if args.balance_classes and not hasattr(model, 'class_weight'):
                sample_weights = np.zeros_like(Y_train)
                for class_value in np.unique(Y_train):
                    mask = Y_train == class_value
                    count = mask.sum()
                    weight = len(Y_train) / float(count)
                    print " -- sample weight for %d = %0.4f" % (
                        class_value, weight)
                    sample_weights[mask] = weight
                model.fit(X_train, Y_train, sample_weight = sample_weights)
            else:
                # either there's no class balancing or it's been
                # handled by the model's constructor
                model.fit(X_train, Y_train)

            pred = model.predict(X_test)
            accuracy = np.mean(pred == Y_test)

            # some classifier models come with a continuous 'decision_function',
            # for those that don't use the probability of the positive class
            if hasattr(model, 'decision_function'):
                prob = model.decision_function(X_test)
            else:
                prob = model.predict_proba(X_test)[:,-1]

            auc = roc_auc_score(Y_test, prob)
            print " * %s\n -- Accuracy=%0.4f, AUC=%0.4f)" % (model, accuracy, auc)
            if auc > best_auc:
                best_model = model
                best_accuracy = accuracy
                best_auc = auc

        if hasattr(best_model, 'coef_'):
            coeff = best_model.coef_.ravel()
        else:
            coeff = best_model.feature_importances_

        abs_coeff = np.abs(coeff)
        nz_coeff_mask = abs_coeff > args.feature_importance_cutoff
        nnz_coeff = nz_coeff_mask.sum()
        prct_nz_coeff = nz_coeff_mask.mean()
        print "-- Fraction nonzero coeffs: %0.4f" % prct_nz_coeff
        if prct_nz_coeff == 0:
            print "Skipping iteration %d due to all zero features" % (i+1)
            continue
        if best_auc < 0.5:
            print "Skipping iteration #%d due to low AUC: %0.4f" % (
                i+1, best_auc
            )
            continue

        # value of a predictor is how much better than baseline it did
        diff = (best_auc - 0.5)
        print "-- Improvement over baseline: %0.4f" % diff
        value = diff / baseline_auc
        total = abs_coeff.sum()
        fractions = abs_coeff / total
        for i, p in enumerate(fractions):
            feature_idx = feature_indices[i]
            name = feature_names[feature_idx]
            feature_counts[name] += 1
            feature_nonzero[name] += value * p

    feature_scores = collections.Counter()
    for name, v in feature_nonzero.iteritems():
        feature_scores[name] = v / float(feature_counts[name])

    keep = []
    n_useful = 0
    for name, acc in feature_scores.most_common()[::-1]:
        print name, acc, "(%d)" % feature_counts[name]

    n_kept = int(args.keep_feature_fraction * n_features)
    for name, acc in feature_scores.most_common(n_kept):
        if acc >  0:
            col = f[name][:]
            keep.append(col)
    print "---"
    print "# useful features: %d / %d" % (len(keep), n_features)

    X_kept = np.array(keep).T
    print "Final X.shape", X_kept.shape
    output_dictionary = {"X": X_kept, "y":y}
    for attr_name in sample_attribute_names:
        output_dictionary[attr_name] = f[attr_name][:]
    np.savez(args.output_data_file, **output_dictionary)