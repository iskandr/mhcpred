import argparse

from parsing import  parse_fasta_mhc_files

import h5py
from pepdata import amino_acid, pmbec
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description=
        """
        Create pairwise amino acid features between peptides and MHC seqs
        """
)
parser.add_argument(
    "--peptide-length",
    default=9,
    type=int,
)
parser.add_argument(
    "--limit-num-alleles",
    default=None,
    type=int,
    help="Only transform a subset of the MHC alleles (for debugging)"
)
parser.add_argument(
    "--mhc-binding-file",
    required=True,
    help="CSV file contaning peptide-MHC binding affinities"
)
parser.add_argument(
    "--mhc-binding-sep",
    default="\t",
    help="Separator used in MHC binding data file"
)
parser.add_argument(
    "--mhc-binding-measurement-column",
    default="meas",
    help="Name of IC50 measurement column in binding data file"
)
parser.add_argument(
    "--mhc-binding-allele-column",
    default="mhc",
    help="Name of MHC allele column in binding data file"
)

parser.add_argument(
    "--mhc-binding-peptide-column",
    default="sequence",
    help="Name of peptide sequence column in binding data file"
)

parser.add_argument(
    "--mhc-binding-affinity-inequality-column",
    default="inequality",
    help="Name of column in binding data file which says whether IC50 = or >"
)

parser.add_argument(
    "--mhc-seqs-file",
    required=True,
    help="FASTA file with MHC pseudosequences"
)

parser.add_argument(
    "--output-file",
    default="pairwise_features.hdf",
    help="Output HDF5 file"
)

parser.add_argument(
    "--output-allele-column",
    default=None,
    help="Name of MHC allele column in output file"
)

parser.add_argument(
    "--min-feature-variance",
    default=10.0**-6,
    type=float,
    help="Smallest variance in a feature for us to keep it",
)

AA_FEATURES = [
    'hydropathy',
    'volume',
    'polarity',
    'prct_exposed_residues',
    'hydrophilicity',
    'accessible_surface_area',
    'local_flexibility',
    'refractivity',
    'alpha_helix_score',
    'beta_sheet_score',
    'turn_score'
]

PAIRWISE_FEATURES = [
    'coil_vs_strand',
    'helix_vs_strand',
    'coil_vs_helix',
    'blosum50',
]


for name in AA_FEATURES + PAIRWISE_FEATURES:
    assert hasattr(amino_acid, name), name



if __name__ == "__main__":
    args = parser.parse_args()
    df_peptides = pd.read_csv(
        args.mhc_binding_file,
        sep=args.mhc_binding_sep)
    print "Loaded %d peptide/allele entries" % len(df_peptides)
    print df_peptides.columns


    lengths = df_peptides[args.mhc_binding_peptide_column].str.len()
    length_mask =  lengths == args.peptide_length
    original_rowcount = len(df_peptides)
    df_peptides = df_peptides[length_mask]

    print
    print "Restricting length to %d: %d / %d rows" % (
        args.peptide_length, len(df_peptides), original_rowcount
    )

    binding_alleles = df_peptides[args.mhc_binding_allele_column]
    mhc_seqs = parse_fasta_mhc_files([args.mhc_seqs_file])

    print
    print "Missing allele sequences:", \
        set(binding_alleles).difference(mhc_seqs.keys())


    seq_pairs = []
    Y = []
    alleles = []
    distinct_allele_count = 0

    for allele in sorted(mhc_seqs.keys()):
        if args.limit_num_alleles:
            if distinct_allele_count >= args.limit_num_alleles:
                break

        mhc_seq = mhc_seqs[allele]

        mask = binding_alleles == allele
        mask |= binding_alleles == allele.replace(":", "")
        subset = df_peptides[mask]
        count = len(subset)
        if count > 0:
            print allele, count
            distinct_allele_count += 1
            peptides = subset[args.mhc_binding_peptide_column]
            ic50s = subset[args.mhc_binding_measurement_column]
            for peptide, ic50 in zip(peptides, ic50s):
                seq_pairs.append((mhc_seq, peptide))
                Y.append(ic50)
                alleles.append(allele)
    print "Total # of peptide pairs: %d" % len(seq_pairs)

    f = h5py.File(args.output_file, 'w')
    Y = np.array(Y)
    f['Y'] = Y
    f['Y_binary'] = Y <= 500
    Y_cat = np.zeros_like(Y, dtype=int)
    Y_cat[Y < 50] = 3
    Y_cat[(Y >= 50) & (Y < 500)] = 2
    Y_cat[(Y >= 500) & (Y < 5000)] = 1
    f['Y_cat'] = Y_cat

    output_mhc_column_name = args.output_allele_column
    if not output_mhc_column_name:
        # if an output column name isn't specified for MHC alleles,
        # use the same name as the input file
        output_mhc_column_name = args.mhc_binding_allele_column
    f[output_mhc_column_name] = alleles

    mhc_seq, pep_seq =  seq_pairs[0]
    mhc_len = len(mhc_seq)
    pep_len = len(pep_seq)

    def add_feature(colname, vec):
        vec = np.array(vec)
        std = np.std(vec)
        if std < args.min_feature_variance:
            print "-- Insufficient variance in feature %s (%0.4f)" % (
                colname, std)
        else:
            print colname
            f[colname] = vec

    def add_single_residue_features(d, name):
        for i in xrange(pep_len):
            vec = []
            colname = name + "_pep_%d" % i
            for (_, pep_seq) in seq_pairs:
                vec.append(d[pep_seq[i]])
            add_feature(colname, vec)

        for i in xrange(mhc_len):
            vec = []
            colname = name + "_mhc_%d" % i
            for (mhc_seq, _) in seq_pairs:
                vec.append(d[mhc_seq[i]])
            add_feature(colname, vec)

    def add_single_residue_features(d, name):
        for i in xrange(pep_len):
            vec = []
            colname = name + "_pep_%d" % i
            for (_, pep_seq) in seq_pairs:
                vec.append(d[pep_seq[i]])
            add_feature(colname, vec)

        for i in xrange(mhc_len):
            vec = []
            colname = name + "_mhc_%d" % i
            for (mhc_seq, _) in seq_pairs:
                vec.append(d[mhc_seq[i]])
            add_feature(colname, vec)

    for name in AA_FEATURES:
        d = getattr(amino_acid, name)
        # some of these are dictionaries, other are SequenceTransformer objects
        if hasattr(d, 'value_dict'):
            d = d.value_dict
        add_single_residue_features(d, name)


    def add_pairwise_features(d, name):
      for i in xrange(mhc_len):
        for j in xrange(pep_len):
            colname = "%s_mhc_%d_pep_%d" % (name, i, j)
            vec = []
            for (mhc_seq, pep_seq) in seq_pairs:
                x = mhc_seq[i]
                y = mhc_seq[j]
                vec.append(d[x][y])
            add_feature(colname, vec)

    def add_neighboring_mhc_features(d, name):
        for i in xrange(0, mhc_len-1):
            for j in [i+1]:
                colname = "%s_mhc_%d_mhc_%d" % (name, i, j)
                vec = []
                for (mhc_seq, _) in seq_pairs:
                    vec.append(d[mhc_seq[i]][mhc_seq[j]])
                add_feature(colname, vec)

    def add_neighboring_pep_features(d, name):
        for i in xrange(0, pep_len-1):
            for j in [i+1]:
                colname = "%s_pep_%d_pep_%d" % (name, i, j)
                vec = []
                for (_, pep_seq) in seq_pairs:
                    vec.append(d[pep_seq[i]][pep_seq[j]])
                add_feature(colname, vec)

    def add_matrix_row_features(d, name):
        """
        Encode amino acids of MHC and peptide using whole rows of
        pairwise coefficient matrix
        """
        assert len(d) > 0

        rows = {}
        keys = d.keys()

        for key1 in keys:
            row_dict = d[key1]
            row = []
            for key2 in keys:
                v = row_dict[key2]
                row.append(row_dict[key2])
            rows[key1] = row

        assert len(rows) > 0
        row_len = len(rows.values()[0])
        assert row_len > 0
        assert all(len(row) == row_len for row in rows.values()), \
            "Lengths: %s != %d" % (
                [ (k, len(rows[k])) for k in rows.keys()],
                row_len
            )

        for i in xrange(pep_len):
            for j in xrange(row_len):
                colname = "%s_pep_%d_row_%d" % (name, i, j)
                vec = []
                for _, pep_seq in seq_pairs:
                    residue = pep_seq[i]
                    row = rows[residue]
                    vec.append(row[j])
                add_feature(colname, vec)

        for i in xrange(mhc_len):
            for j in xrange(row_len):
                colname = "%s_mhc_%d_row_%d" % (name, i, j)
                vec = []
                for mhc_seq, _ in seq_pairs:
                    mhc_residue = mhc_seq[i]
                    row = rows[mhc_residue]
                    vec.append(row[j])
                add_feature(colname, vec)

    matrices = [(name, getattr(amino_acid, name)) for name in PAIRWISE_FEATURES]
    pmbec_dict = pmbec.read_coefficients(key_type='row')
    matrices.append( ('pmbec', pmbec_dict) )

    for name, d in matrices:
        add_pairwise_features(d, name)
        add_neighboring_mhc_features(d, name)
        add_neighboring_pep_features(d, name)
        add_matrix_row_features(d, name)

    print "Generated %d features" % len(f.keys())
    print "Closing file %s..." % args.output_file
    f.close()