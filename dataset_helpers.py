import numpy as np

def extract_columns(df_peptides):
    ic50 = np.array(df_peptides["IC50_Median"])
    has_ic50 = np.array(df_peptides["IC50_Count"] > 0)
    ic50_in_range = np.array(df_peptides["IC50_Median"] < 10**7)
    ic50_mask = has_ic50 & ic50_in_range
    ic50[~ic50_mask] = 0

    ic50_high = (ic50 > 0) & (ic50 <= 50)
    ic50_mid = (ic50 > 50) & (ic50 <= 500)
    ic50_low = (ic50 > 500) & (ic50 <= 2500)

    print "Keeping %d entries for IC50 data" % ic50_mask.sum()

    high = df_peptides['Positive-High'] + ic50_high 
    mid = df_peptides["Positive-Intermediate"]  + ic50_mid 
    low = df_peptides['Positive-Low']   + ic50_low 
    pos_unknown = df_peptides['Positive']
    
    # spread probability over multiple binding ranges
    # if peptide wasn't marked with a L/M/H level
    high += 0.35 * pos_unknown
    mid += 0.4 * pos_unknown
    low += 0.25 * pos_unknown

    # we want to discretize strong binders, low binders, and non-binders 
    neg = df_peptides["Negative"]

    neg_mask = (neg >= low) & (neg >= mid) & (neg >= high)
    low_mask = ~neg_mask & (low >= mid) & (low >= high)
    mid_mask = ~neg_mask & ~low_mask & (mid >= high)
    high_mask = ~neg_mask & ~low_mask & ~mid_mask


    category = np.zeros(len(high), dtype=int)
    print category.shape, category.dtype
    print neg_mask.shape, neg_mask.dtype

    category[np.array(neg_mask)] = -1
    category[np.array(mid_mask)] = 1
    category[np.array(high_mask)] = 2



    print "# binding category == 2", (category == 2).sum()
    print "# binding category == 1", (category == 1).sum()
    print "# binding category == 0", (category == 0).sum()
    print "# binding category == -1", (category == -1).sum()
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