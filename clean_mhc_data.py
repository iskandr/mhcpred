import pandas as pd
df = pd.read_excel('mhc.xlsx')

print "Initial size", len(df)

alleles = df['MHC Allele']

# drop missing allele names
mask = ~alleles.isnull()
# only count human HLA types 
mask &= (alleles.str.startswith("HLA-") | alleles.str.startswith("Patr-"))
# drop 2-digit partial HLA types like HLA-A2 and HLA-DR
mask &= alleles.str.len() > 7

binding_scores = df['Quantitative Result']

# must have a binding score
mask &= ~binding_scores.isnull()

epitope_type = df['Epitope Type']
print "Epitope types", df['Epitope Type'].unique()
mask &=  epitope_type == 'Linear peptide'

assay_units = df['Assay Units']
mask &= assay_units == 'IC50 nM'

assay_method = df['Assay Method']
print "Assay methods", df['Assay Method'].unique()
mask &= assay_method.str.startswith("Purified MHC")


epitope = df['Epitope']
mask &= ~epitope.str.contains('\(|\)|\+')

df_subset = df[mask]

df_clean = pd.DataFrame({
	'MHC Allele' : df_subset['MHC Allele'], 
	'Epitope' : df_subset['Epitope'].str.upper().str.strip(), 
	'IC50': df_subset['Quantitative Result'],
})

type1_mask = df_clean['MHC Allele'].str.contains("(HLA|Patr)-(A|B|C)")

df_type1 = df_clean[type1_mask]
print "# Type 1", len(df_type1)
df_type1.to_csv("mhc1.csv")

type2_mask = df_clean['MHC Allele'].str.startswith("HLA-D")
df_type2 = df_clean[type2_mask]
print "# Type 2", len(df_type2)
df_type2.to_csv("mhc2.csv")

print "# ignored", len(df_clean) - len(df_type1) - len(df_type2)
