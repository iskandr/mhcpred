from collections import OrderedDict

import pandas as pd 
import numpy as np 

from os.path import exists 
import subprocess
import pandas as pd
import logging 
from epitopes import amino_acid, reduced_alphabet

from dataset_helpers import filter_dataframe, extract_columns 
        
def generate_aa_feature_data(
        df_peptides,
        df_mhc, 
        neighboring_residue_interactions=False,
        length = 9):

    
    peptide_alleles = df_peptides['MHC Allele'].str.replace("*", '')
    peptide_seqs = df_peptides['Epitope']
    ic50 = np.array(df_peptides["IC50"])
    category = np.zeros(len(ic50), dtype=int)
    category[ic50 >= 5000] = -1
    category[ic50 < 5000] = 0
    category[ic50 <= 500] = 1
    category[ic50 <= 50] = 2

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
    X_solo =[]
    Y_IC50 = []
    Y_category = []
    alleles = []
    n_dims = length * len(mhc_seqs[0])
    
    features = [
        amino_acid.volume.value_dict, 
        amino_acid.hydropathy.value_dict,
        amino_acid.local_flexibility.value_dict,
        
        amino_acid.polarity.value_dict,
        amino_acid.prct_exposed_residues.value_dict,
        amino_acid.alpha_helix_score_dict,
        amino_acid.beta_sheet_score_dict,
        amino_acid.turn_score_dict,
        # Often end up with 0 coefficients in L1-regularized classifiers
        #amino_acid.accessible_surface_area.value_dict,
        #amino_acid.pK_side_chain.value_dict,
        
    ]
    n_skipped = 0
    for peptide_idx, allele in enumerate(peptide_alleles):

        if allele in mhc_seqs_dict:
            allele_seq = mhc_seqs_dict[allele]
            peptide = peptide_seqs.ix[peptide_idx]
            if len(peptide) != length:
                print "-- skipping %s, length = %d" % (
                    peptide, len(peptide)
                )
                n_skipped += 1
                continue 
            
            n_peptide_letters = len(peptide)
            n_mhc_letters = len(allele_seq)
            curr_ic50 = ic50[peptide_idx] 
            binder = category[peptide_idx]
            print peptide_idx, allele, peptide, curr_ic50, binder
           
            pep_vec = []
            mhc_vec = []
            for f in features:
                pep_vec.extend([f[x] for x in peptide])
                mhc_vec.extend([f[x] for x in allele_seq])
            
            vec = pep_vec + mhc_vec
            
            X.append(np.array(vec))
            X_solo.append(np.array(pep_vec))
            Y_IC50.append(curr_ic50)
            Y_category.append(binder)
            alleles.append(allele)
    print "Skipped %d peptides" % n_skipped
    X = np.array(X)
    X_solo = np.array(X_solo)
    assert X.ndim == 2, "Expected two-dimensional X, got %s" % (X.shape,)
    Y_IC50 = np.array(Y_IC50)
    Y_category = np.array(Y_category)
    assert Y_IC50.ndim == 1, "Expected Y_IC50 to be a vector"
    assert Y_category.ndim == 1, "Expected Y_category to be a vector"
    assert len(Y_IC50) == X.shape[0]
    assert len(Y_category) == X.shape[0]
    return X, X_solo, Y_IC50, Y_category, alleles



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(
        description=
            """
            Create pairwise interaction index features
            and labels from MHC binding data
            """.replace("\n", ' ')
    )

    parser.add_argument(
        "--input-file", 
        default='iedb_public_mhc_data_2013.txt')

    parser.add_argument(
        "--mhc-seq-filename", 
        default = "MHC_aa_seqs.csv")

    """
    parser.add_argument(
        "--length",
        type=int, 
        default=9)

    parser.add_argument("--human", 
        default = False, 
        action="store_true")
    """
    parser.add_argument(
        "--mhc-class",
        default="I")

    args = parser.parse_args()
    df_raw = pd.read_csv(args.input_file, sep = "\t")
    df_peptides = pd.DataFrame()
    df_peptides["MHC Allele"] = df_raw['mhc']
    df_peptides["Epitope"] = df_raw['sequence']
    df_peptides["IC50"] = df_raw['meas']


    print "Loaded %d entries" % len(df_peptides)
    
    df_mhc = pd.read_csv(args.mhc_seq_filename)
    print df_mhc
    print "Loaded %d MHC alleles" % len(df_mhc)
    X, X_pep, Y_IC50, Y_category, alleles = generate_aa_feature_data(
        df_peptides, 
        df_mhc)

    print "Generated X.shape = %s" % (X.shape,)
    print "# IC50 target values = %d" % (Y_IC50> 0).sum()
    print "Saving to disk..."
    np.save("aa_features_X.npy", X)
    np.save("aa_features_X_pep.npy", X_pep)
    np.save("aa_features_Y_IC50.npy", Y_IC50)
    np.save("aa_features_Y_category.npy", Y_category)
    with open('aa_features_alleles.txt', 'w') as f:
        for allele in alleles:
            f.write(allele)
            f.write("\n")
