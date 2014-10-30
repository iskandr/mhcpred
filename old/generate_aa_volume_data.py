from collections import OrderedDict

import pandas as pd 
import numpy as np 

from os.path import exists 
import subprocess
import pandas as pd
import logging 
from epitopes.amino_acid import volume 

from dataset_helpers import filter_dataframe, extract_columns 
        
def generate_pairwise_index_data(
        df_peptides,
        df_mhc, 
        neighboring_residue_interactions=False):

    (peptide_alleles, peptide_seqs, category, ic50, ic50_mask) = \
        extract_columns(df_peptides)
    print "%d unique peptide alleles" % len(peptide_alleles.unique())
    
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
    
    Volume = volume.value_dict 

    for peptide_idx, allele in enumerate(peptide_alleles):

        if allele in mhc_seqs_dict:
            allele_seq = mhc_seqs_dict[allele]
            peptide = peptide_seqs.ix[peptide_idx]
            
            n_peptide_letters = len(peptide)
            n_mhc_letters = len(allele_seq)
            curr_ic50 = ic50[peptide_idx] * ic50_mask[peptide_idx]
            binder = category[peptide_idx]
            print peptide_idx, allele, peptide, curr_ic50, binder
            
            #print peptide_idx, allele, peptide, allele_seq, ic50
            vec_pep = [Volume[x] for x in peptide]
            vec_contact = [Volume[x] for x in allele_seq]
            vec = vec_pep + vec_contact
            assert vec 
            assert len(vec) > 0
            if len(X) > 0:
                assert len(vec) == len(X[-1]), \
                    "Weird vector length %d (expected %d)" % \
                    (len(vec), len(X[-1]))
            X.append(np.array(vec))
            Y_IC50.append(curr_ic50)
            Y_category.append(binder)
            alleles.append(allele)
    X = np.array(X)
    assert X.ndim == 2, "Expected two-dimensional X, got %s" % (X.shape,)
    Y_IC50 = np.array(Y_IC50)
    Y_category = np.array(Y_category)
    assert Y_IC50.ndim == 1, "Expected Y_IC50 to be a vector"
    assert Y_category.ndim == 1, "Expected Y_category to be a vector"
    assert len(Y_IC50) == X.shape[0]
    assert len(Y_category) == X.shape[0]
    return X, Y_IC50, Y_category, alleles



def generate_aa_feature_data(df_peptides, df_mhc):
    alleles, epitopes, category, ic50, ic50_mask = extract_columns(df_peptides)
    for i, epitope in enumerate(epitopes):
        allele = alleles.irow(i)

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
        default='mhc_grouped.csv')

    parser.add_argument(
        "--mhc-seq-filename", 
        default = "MHC_aa_seqs.csv")

 
    parser.add_argument(
        "--length",
        type=int, 
        default=9)

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
    df_peptides = filter_dataframe(
        df_peptides,
        mhc_class = args.mhc_class, 
        length = args.length, 
        human = args.human)

    df_mhc = pd.read_csv(args.mhc_seq_filename)
    print df_mhc
    print "Loaded %d MHC alleles" % len(df_mhc)
    X, Y_IC50, Y_category, alleles = generate_pairwise_index_data(
        df_peptides, 
        df_mhc)

    print "Generated X.shape = %s" % (X.shape,)
    print "# IC50 target values = %d" % (Y_IC50> 0).sum()
    print "# binding category values = %d" % (Y_category!=0).sum()
    print "Saving to disk..."
    np.save("volume_X.npy", X)
    np.save("volume_Y_IC50.npy", Y_IC50)
    np.save("volume_Y_category.npy", Y_category)
    with open('volume_alleles.txt', 'w') as f:
        for allele in alleles:
            f.write(allele)
            f.write("\n")
