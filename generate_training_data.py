from collections import OrderedDict

import pandas as pd 
import numpy as np 
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS

from os.path import exists 
import subprocess
import pandas as pd
import logging 

        
def generate_training_data(
        df_peptides,
        df_mhc, 
        neighboring_residue_interactions = False,
        mhc_class = "I",
        length = 9):

    df_peptides['MHC Allele'] = \
        df_peptides['MHC Allele'].str.replace('*', '').str.strip()
    
    if length:
        df_peptides = df_peptides[df_peptides.Epitope.str.len() == 9]
    
    if mhc_class:
        df_peptides = df_peptides[df_peptides['MHC Class'] == mhc_class]
    
    has_ic50 = df_peptides["IC50_Count"] > 0
    ic50_in_range = df_peptides["IC50_Median"] < 10**7
    ic50_mask = has_ic50 & ic50_in_range

    print "Keeping %d entries for IC50 data" % ic50_mask.sum()

    pos_count = (df_peptides["Positive"] + df_peptides["Positive-High"]) 
    pos_count += 0.5 * df_peptides["Positive-Intermediate"]
    pos_count += 0.25 * df_peptides["Positive-Low"]
    
    neg_count = df_peptides["Negative"]

    diff = (pos_count - neg_count)
    pos_mask = diff >= 1
    neg_mask = diff <= -1

    category_mask = pos_mask | neg_mask
    print "Keeping %d entries for categorical data" % category_mask.sum()
    category = np.sign(diff)

    assert False

    # reformat HLA allales 'HLA-A*03:01' into 'HLA-A03:01'
    peptide_alleles = df_peptides['MHC Allele']
    peptide_seqs = df_peptides['Epitope']
    peptide_ic50 = df_peptides['IC50']
    
    print "%d unique peptide alleles" % len(peptide_alleles.unique())
    

    mhc_alleles = df_mhc['Allele'].str.replace('*', '')
    mhc_seqs = df_mhc['Residues']

    assert len(mhc_alleles) == len(df_mhc)
    assert len(mhc_seqs) == len(df_mhc)

    print list(sorted(peptide_alleles.unique()))
    logging.info(
        "%d common alleles",
        len(set(mhc_alleles).intersection(set(peptide_alleles.unique())))
    )
    logging.info(
        "Missing allele sequences for %s", 
        set(peptide_alleles.unique()).difference(set(mhc_alleles))
    )

    mhc_seqs_dict = {}
    for allele, seq in zip(mhc_alleles, mhc_seqs):
        mhc_seqs_dict[allele] = seq 


    X = []
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
            #print peptide_idx, allele, peptide, allele_seq, ic50
            vec = [AMINO_ACID_PAIR_POSITIONS[peptide_letter + mhc_letter] 
                    for peptide_letter in peptide
                    for mhc_letter in allele_seq]

            if neighboring_residue_interactions:
                # add interaction terms for neighboring residues on the peptide
                for i, peptide_letter in enumerate(peptide):
                    if i > 0:
                        before = peptide_substring[i - 1]
                        vec.append(
                            AMINO_ACID_PAIR_POSITIONS[before + peptide_letter]
                        )
                    if i < 8:
                        after = peptide_substring[i + 1]
                        vec.append(
                            AMINO_ACID_PAIR_POSITIONS[peptide_letter + after]
                        )
                
            X.append(np.array(vec))
            Y.append(ic50)
            alleles.append(allele)
    X = np.array(X)
    Y = np.array(Y)

    print "IC50 min = %f, max = %f, median = %f" % \
        (np.min(Y),
         np.max(Y),
         np.median(Y)
        )
    print "Generated data shape", X.shape
    assert len(Y) == X.shape[0]
    return X, Y, alleles



def save_training_data(X, Y, alleles):
    print "Saving to disk..."
    np.save("X.npy", X)
    np.save("Y.npy", Y)
    with open('alleles.txt', 'w') as f:
        for allele in alleles:
            f.write(allele)
            f.write("\n")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(
        description='Create feature vectors and labels from MHC binding data')

    parser.add_argument(
        "--input-file", 
        default='mhc_grouped.csv')

    parser.add_argument(
        "--mhc-seq-filename", 
        default = "MHC_aa_seqs.csv")

    parser.add_argument(
        "--neighboring-residues", 
        action='store_true', 
        default=False)

    parser.add_argument(
        "--length",
        type=int, 
        default=9)

    parser.add_argument(
        "--mhc-class",
        default="I")
       
    args = parser.parse_args()
    df_peptides = pd.read_csv(args.input_file)
    print "Loaded %d peptide/allele entries", len(df_peptides)
    print df_peptides.columns
    
    df_mhc = pd.read_csv(args.mhc_seq_filename)
    print "Loaded %d MHC alleles" % len(df_mhc)

    X,Y,alleles = generate_training_data(df_peptides, df_mhc)
    save_training_data(X, Y, alleles)

