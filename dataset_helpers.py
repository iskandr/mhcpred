import numpy as np
def extract_columns(df_peptides):
    ic50 = np.array(df_peptides["IC50_Median"])
    has_ic50 = np.array(df_peptides["IC50_Count"] > 0)
    ic50_in_range = np.array(df_peptides["IC50_Median"] < 10**7)
    ic50_mask = has_ic50 & ic50_in_range
    ic50[~ic50_mask] = 0

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