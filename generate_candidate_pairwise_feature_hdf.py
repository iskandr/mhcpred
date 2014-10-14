import argparse

from parsing import  parse_fasta_mhc_files

import h5py

from epitopes import amino_acid

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
    help="FASTA file with MHC pseudosequences")

parser.add_argument(
    "--output-file",
    default="pairwise_features.hdf",
    help="Output HDF5 file"
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
    'alpha_helix_score_dict',
    'beta_sheet_score_dict',
    'turn_score_dict'
]

PAIRWISE_FEATURES = [
    'coil_vs_strand_dict',
    'helix_vs_strand_dict',
    'coil_vs_helix_dict',
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
    binding_alleles = df_peptides[args.mhc_binding_allele_column]
    mhc_seqs = parse_fasta_mhc_files([args.mhc_seqs_file])

    print
    print "Missing allele sequences:", \
        set(binding_alleles).difference(mhc_seqs.keys())

    lengths = df_peptides[args.mhc_binding_peptide_column].str.len()
    length_mask =  lengths == args.peptide_length
    original_rowcount = len(df_peptides)
    df_peptides = df_peptides[length_mask]

    print
    print "Restricting length to %d: %d / %d rows" % (
        args.peptide_length, len(df_peptides), original_rowcount
    )

    seq_pairs = []
    Y = []
    for allele in sorted(mhc_seqs.keys()):
        mhc_seq = mhc_seqs[allele]

        mask = binding_alleles == allele
        mask |= binding_alleles == allele.replace(":", "")
        subset = df_peptides[mask]
        count = len(subset)
        if count > 0:
            print allele, count
            peptides = subset[args.mhc_binding_peptide_column]
            ic50s = subset[args.mhc_binding_measurement_column]
            for peptide, ic50 in zip(peptides, ic50s):
                seq_pairs.append((mhc_seq, peptide))
                Y.append(ic50)
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

    mhc_seq, pep_seq =  seq_pairs[0]
    mhc_len = len(mhc_seq)
    pep_len = len(pep_seq)
    for name in AA_FEATURES:

        d = getattr(amino_acid, name)
        # some of these are dictionaries, other are SequenceTransformer objects
        if hasattr(d, 'value_dict'):
            d = d.value_dict

        for i in xrange(pep_len):
            vec = []
            colname = name + "_pep_%d" % i
            for (_, pep_seq) in seq_pairs:
                vec.append(d[pep_seq[i]])
            f[colname] = np.array(vec)

        for i in xrange(mhc_len):
            vec = []
            colname = name + "_mhc_%d" % i
            print colname
            for (mhc_seq, _) in seq_pairs:
                vec.append(d[mhc_seq[i]])
            f[colname] = np.array(vec)

    for name in PAIRWISE_FEATURES:
        d = getattr(amino_acid, name)

        for i in xrange(mhc_len):
            for j in xrange(pep_len):
                colname = "%s_mhc_%d_pep_%d" % (name, i, j)
                print colname
                vec = []
                for (mhc_seq, pep_seq) in seq_pairs:
                    x = mhc_seq[i]
                    y = mhc_seq[j]
                    vec.append(d[x][y])
                f[colname] = np.array(vec)
    print "Closing file..."
    f.close()