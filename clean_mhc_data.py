
from os.path import exists 
import subprocess
import pandas as pd
only_human = False 

if __name__ == '__main__':
    if not exists("mhc_ligand_full.csv"):
        subprocess.check_call(["wget","http://www.iedb.org/doc/mhc_ligand_full.zip"])
        subprocess.check_call(["unzip", "mhc_ligand_full.zip"])
    df = pd.read_csv('mhc_ligand_full.csv', error_bad_lines=False, header=[0,1])
    print "Initial size", len(df)
    epitopes = df['Epitope']['Description'].str.upper().str.strip()
    
    # make sure there is an epitope string and it's at least a 5mer
    mask = ~epitopes.isnull()
    mask &= epitopes.str.len() >= 5 
    # drop epitopes with special characters
    mask &= ~epitopes.str.contains('\(|\)|\+')
    alleles = df['MHC']['Allele Name']
    mhc_class =  df['MHC']['MHC allele class']
    
    # drop missing allele names
    mask &= ~alleles.isnull()
    # drop 2-digit partial HLA types like HLA-A2 and HLA-DR
    mask &= alleles.str.len() > 7
    if only_human:
        # only count human HLA types 
        mask &= alleles.str.startswith("HLA-") 
  
    epitope_type = df['Epitope']['Object Type']
    print "Epitope types", epitope_type.unique()
    mask &=  epitope_type == 'Linear peptide'
    
    categories = df['Assay'][['Qualitative measurement']]
    category_mask = ~categories.isnull()
    binding_scores = df['Assay']['Quantitative measurement']
    assay_units = df['Assay']['Units']
    mask &= assay_units.str.startswith('IC50')
    assay_method = df['Assay']['Method/Technique']
    df_clean = pd.DataFrame({
    	'MHC Allele' : alleles[mask], 
    	'Epitope' : epitopes[mask], 
    	'IC50': binding_scores[mask],
    })

    df_clean.to_csv("mhc.csv")

    type1_mask = (mhc_class == "I")[mask]

    df_type1 = df_clean[type1_mask]
    print "# Type 1", len(df_type1)
    df_type1.to_csv("mhc1.csv")

    type2_mask = (mhc_class == "II")[mask]
    df_type2 = df_clean[type2_mask]
    print "# Type 2", len(df_type2)
    df_type2.to_csv("mhc2.csv")

    print "# ignored", len(df_clean) - len(df_type1) - len(df_type2)
