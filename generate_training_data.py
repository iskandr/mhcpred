import pandas as pd 
import numpy as np 
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS


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
    alleles = []
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
                alleles.append(allele)
    X = np.array(X)
    W = np.array(W)
    Y = np.array(Y)

    print "Generated data shape", X.shape
    assert len(W) == X.shape[0]
    assert len(Y) == X.shape[0]
    return X, W, Y, alleles


