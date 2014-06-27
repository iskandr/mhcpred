import pandas as pd
df = pd.read_excel('mhc.xlsx')

print "Initial size", len(df)

alleles = df['MHC Allele']

# drop missing allele names
mask = ~alleles.isnull()
# only count human HLA types 
mask &= alleles.str.startswith("HLA-")
# drop 2-digit partial HLA types like HLA-A2 and HLA-DR
mask &= alleles.str.len() > 7

binding_scores = df['Quantitative Result']

# must have a binding score
mask &= ~binding_scores.isnull()

epitope_type = df['Epitope Type']
mask &=  epitope_type == 'Linear peptide'

assay_units = df['Assay Units']
mask &= assay_units == 'IC50 nM'

assay_method = df['Assay Method']
mask &= assay_method.str.startswith("Purified MHC")

df_subset = df[mask]

df_clean = pd.DataFrame({
	'MHC Allele' : df_subset['MHC Allele'], 
	'Epitope' : df_subset['Epitope'], 
	'IC50': df_subset['Quantitative Result'],
})

type1_mask = df_clean['MHC Allele'].str.contains("HLA-(A|B|C)")

df_type1 = df_clean[type1_mask]
print "# Type 1", len(df_type1)
df_type1.to_csv("mhc1.csv")

type2_mask = df_clean['MHC Allele'].str.startswith("HLA-D")
df_type2 = df_clean[type2_mask]
print "# Type 2", len(df_type2)
df_type2.to_csv("mhc2.csv")

