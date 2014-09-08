import pandas as pd 
import numpy as np 
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS


from os.path import exists 
import subprocess
import pandas as pd
import logging 


IEDB_FILENAME = "mhc_ligand_full.csv"
IEDB_URL = "http://www.iedb.org/doc/mhc_ligand_full.zip"

def download_iedb_database(filename, url):
    if not exists(filename):
        subprocess.check_call(["wget", url])
        zipped_name = url.split("/")[-1]
        subprocess.check_call(["unzip", zipped_name])
    return pd.read_csv(filename, error_bad_lines=False, header=[0,1])
    

def load_iedb(filename, url, only_human = False):

    df = download_iedb_database(filename, url)
    
    print "IEDB contains %d entries" % len(df)

    epitopes = df['Epitope']['Description'].str.upper().str.strip()
    # make sure there is an epitope string and it's at least a 5mer
    mask = ~epitopes.isnull()
    mask &= epitopes.str.len() >= 5 
    
    # must contain only the 20 canonical amino acid letters
    mask = ~epitopes.str.contains("X|B|Z")
    # drop epitopes with special characters
    mask &= ~epitopes.str.contains('\(|\)|\+')

    print "Dropped %d invalid epitope strings" % (len(mask) - mask.sum())

    alleles = df['MHC']['Allele Name']
    
    # drop missing allele names
    mask &= ~alleles.isnull()
    # drop 2-digit partial HLA types like HLA-A2 and HLA-DR
    mask &= alleles.str.len() > 7
    


    if only_human:
        # only count human HLA types 
        mask &= alleles.str.startswith("HLA-") 
  
    epitope_type = df['Epitope']['Object Type']

    mask &=  epitope_type == 'Linear peptide'
    
    df = df[mask]
    epitopes = epitopes[mask]

    mhc_class =  df['MHC']['MHC allele class']
    categories = df['Assay'][['Qualitative Measure']]
    binding_scores = df['Assay']['Quantitative measurement']
    assay_units = df['Assay']['Units']
    assay_method = df['Assay']['Method/Technique']
    paper = df['Reference']['Title']

    df_clean = pd.DataFrame({})
    df_clean['Epitope'] = epitopes
    df_clean['MHC Allele'] = alleles 
    df_clean['MHC Class'] = mhc_class
    df_clean['IC50'] =  binding_scores
    df_clean['Assay Units'] = assay_units
    df_clean['Assay Method'] = assay_method
    df_clean['Binder'] = categories
    df_clean['Paper'] = paper 

    print "Final DataFrame with %d entries" % len(df_clean)
    return df_clean

    
def generate_training_data(
        df_peptides,
        df_mhc, 
        neighboring_residue_interactions = False,
        length = 9):

    
    print "Loaded %d peptides" % len(df_peptides)

    df_peptides['MHC Allele'] = df_peptides['MHC Allele'].str.replace('*', '').str.strip()
    df_peptides['Epitope']  = df_peptides['Epitope'].str.strip().str.upper()

    print "Peptide lengths"
    print df_peptides['Epitope'].str.len().value_counts()


    mask = df_peptides['Epitope'].str.len() == length
    mask &= df_peptides['MHC Class'] == 'I'
    mask &= ~df_peptides['IC50'].isnull()
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
    print mhc_alleles[:20]
    print "%d common alleles" % len(set(mhc_alleles).intersection(set(peptide_alleles.unique())))
    print "Missing allele sequences for %s" % set(peptide_alleles.unique()).difference(set(mhc_alleles))

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

    import argparse
    parser = argparse.ArgumentParser(
        description='Generate training data for MHC binding prediction')

    parser.add_argument(
        "--iedb-filename",
        default = IEDB_FILENAME, 
    )
    parser.add_argument(
        "--iedb-url",
        default = IEDB_URL, 
    )
    parser.add_argument(
        "--iedb-output", 
        default='mhc.csv')

    parser.add_argument(
        "--mhc-seq-filename", 
        default = "MHC_aa_seqs.csv"
    )
    parser.add_argument(
        "--aa-pair-features", 
        default = False, 
        action = "store_true"
    )

    parser.add_argument(
        "--neighboring-residues",  action='store_true', default=False)
       
    args = parser.parse_args()
    
    df_peptides = load_iedb(args.iedb_filename, args.iedb_url)

    df_peptides.to_csv(args.iedb_output, index=False)
    
    if args.aa_pair_features:
        df_mhc = pd.read_csv(args.mhc_seq_filename)
        print "Loaded %d MHC alleles" % len(df_mhc)

        X,Y,alleles = generate_training_data(df_peptides, df_mhc)
        save_training_data(X, Y, alleles)

