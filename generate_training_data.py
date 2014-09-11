from collections import OrderedDict

import pandas as pd 
import numpy as np 
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS

from os.path import exists 
import subprocess
import pandas as pd
import logging 

def extract_columns(df_peptides):
    ic50 = df_peptides["IC50_Median"]
    has_ic50 = df_peptides["IC50_Count"] > 0
    ic50_in_range = df_peptides["IC50_Median"] < 10**7
    ic50_mask = has_ic50 & ic50_in_range

    print "Keeping %d entries for IC50 data" % ic50_mask.sum()

    pos_count = (df_peptides["Positive"] + df_peptides["Positive-High"]) 
    pos_count += 0.5 * df_peptides["Positive-Intermediate"]
    # pos_count = 0.1 * df_peptides["Positive-Low"]
    
    neg_count = df_peptides["Negative"]

    diff = (pos_count - neg_count)
    pos_mask = diff >= 0.5
    neg_mask = diff < -0.5

    category_mask = (pos_mask | neg_mask)
    print "Keeping %d entries for categorical data" % category_mask.sum()
    category = np.sign(diff) * category_mask

    # reformat HLA allales 'HLA-A*03:01' into 'HLA-A03:01'
    alleles = df_peptides['MHC Allele']
    epitopes = df_peptides['Epitope']
    return alleles, epitopes, category, ic50, ic50_mask 
        
def generate_pairwise_index_data(
        df_peptides,
        df_mhc, 
        neighboring_residue_interactions=False):

    
    
    
    print "%d unique peptide alleles" % len(peptide_alleles.unique())
    (peptide_alleles, peptide_seqs, category, ic50, ic50_mask) = \
        extract_columns(df_peptides)

    mhc_alleles = df_mhc['Allele'].str.replace('*', '')
    mhc_seqs = df_mhc['Residues']

    assert len(mhc_alleles) == len(df_mhc)
    assert len(mhc_seqs) == len(df_mhc)

    unique_alleles = set(peptide_alleles.unique())
    print list(sorted(unique_alleles))
    logging.info(
        "%d / %d available alleles",
        len(set(mhc_alleles).intersection(unique_alleles)),
        len(unique_alleles)
    )
    logging.info(
        "Missing allele sequences for %s", 
        list(sorted(unique_alleles.difference(set(mhc_alleles))))
    )

    mhc_seqs_dict = {}
    for allele, seq in zip(mhc_alleles, mhc_seqs):
        mhc_seqs_dict[allele] = seq 
    X = []
    Y_IC50 = []
    Y_category = []
    alleles = []
    n_dims = length * len(mhc_seqs[0])
    for peptide_idx, allele in enumerate(peptide_alleles):

        if allele in mhc_seqs_dict:
            allele_seq = mhc_seqs_dict[allele]
            peptide = peptide_seqs[peptide_idx]
            
            n_peptide_letters = len(peptide)
            n_mhc_letters = len(allele_seq)
            curr_ic50 = ic50[peptide_idx] * ic50_mask[peptide_idx]
            binder = category[peptide_idx]
            print peptide_idx, allele, peptide, curr_ic50, binder
            
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
                    if i < length - 1:
                        after = peptide_substring[i + 1]
                        vec.append(
                            AMINO_ACID_PAIR_POSITIONS[peptide_letter + after]
                        )
                
            X.append(np.array(vec))
            Y_IC50.append(curr_ic50)
            Y_category.append(binder)
            alleles.append(allele)
    X = np.array(X)
    Y_IC50 = np.array(Y_IC50)
    Y_category = np.array(Y_category)
    assert len(Y_IC50) == X.shape[0]
    assert len(Y_category) == X.shape[0]
    return X, Y_IC50, Y_category, alleles

def filter_dataframe(
        df_peptides, 
        human = False, 
        length = None, 
        mhc_class = "I"):

    df_peptides['MHC Allele'] = \
        df_peptides['MHC Allele'].str.replace('*', '').str.strip()
    
    if human:
        human_mask = df_peptides["MHC Allele"].str.startswith("HLA")
        print "Keeping %d / %d  entries for human alleles" % (
            human_mask.sum(),
            len(human_mask),
        )
        df_peptides = df_peptides[human_mask]
        
    if mhc_class:
        mhc_class_mask = df_peptides['MHC Class'] == mhc_class
        print "Keeping %d / %d entries for MHC class = %s" % (
            mhc_class_mask.sum(),
            len(mhc_class_mask),
            mhc_class
        )
        df_peptides = df_peptides[mhc_class_mask]

    if length:
        length_mask = df_peptides.Epitope.str.len() == length
        print "Keeping %d / %d entries with length = %d" % (
            length_mask.sum(), 
            len(length_mask),
            length
        )
        df_peptides = df_peptides[length_mask]
    else:
        df_peptides = df_peptides[df_peptides.Epitope.str.len() > 5]

    df_peptides = df_peptides.reset_index()
    return df_peptides   

def generate_aa_feature_data(df_peptides, df_mhc):
    alleles, epitopes, category, ic50, ic50_mask = extract_columns(df_peptides)
    for i, epitope in enumerate(epitopes):
        allele = alleles.irow(i)

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
        "--pairwise-index-features",
        action = "store_true",
        default = False
    )
    parser.add_argument(
        "--neighboring-residues", 
        action='store_true', 
        default=False)

    parser.add_argument(
        "--length",
        type=int, 
        default=None)

    parser.add_argument("--human", 
        default = False, 
        action="store_true")

    parser.add_argument(
        "--mhc-class",
        default="I")
       
    args = parser.parse_args()
    df_peptides = pd.read_csv(args.input_file)
    print "Loaded %d peptide/allele entries", len(df_peptides)
    print df_peptides.columns
    
    length = args.length
    if not length and args.pairwise_index_features:
        length = 9 
    df_peptides = filter_dataframe(
        df_peptides,
        mhc_class = args.mhc_class, 
        length = args.length, 
        human = args.human)

    df_mhc = pd.read_csv(args.mhc_seq_filename)
    print df_mhc
    print "Loaded %d MHC alleles" % len(df_mhc)

    if args.pairwise_index_features:  
        X, Y_IC50, Y_category, alleles = generate_pairwise_index_data(
            df_peptides, df_mhc)
    else:
        X, Y_IC50, Y_category, alleles = \
            generate_aa_feature_data(df_peptides, df_mhc)

    print "Generated X.shape = %s" % (X.shape,)
    print "# IC50 target values = %d" % (Y_IC50> 0).sum()
    print "# binding category values = %d" % (Y_category!=0).sum()
    print "Saving to disk..."
    np.save("X.npy", X)
    np.save("Y_IC50.npy", Y_IC50)
    np.save("Y_category.npy", Y_category)
    with open('alleles.txt', 'w') as f:
        for allele in alleles:
            f.write(allele)
            f.write("\n")
