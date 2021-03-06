from collections import OrderedDict

import pandas as pd
import numpy as np
from amino_acids import AMINO_ACID_LETTERS, AMINO_ACID_PAIRS, AMINO_ACID_PAIR_POSITIONS

from os.path import exists,  join
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

    logging.info("IEDB contains %d entries", len(df))

    epitopes = df['Epitope']['Description'].str.upper().str.strip()
    # make sure there is an epitope string and it's at least a 5mer
    mask = ~epitopes.isnull()
    mask &= epitopes.str.len() >= 5

    # must contain only the 20 canonical amino acid letters
    mask = ~epitopes.str.contains("X|B|Z", na=False)
    # drop epitopes with special characters
    mask &= ~epitopes.str.contains('\(|\)|\+', na=False)

    logging.info(
        "Dropped %d invalid epitope strings",
        (len(mask) - mask.sum())
    )

    alleles = df['MHC']['Allele Name']

    # drop missing allele names
    mask &= ~alleles.isnull()

    mask &= alleles.str.len() > 5
    mask &= ~alleles.str.contains("Class|class", na=False)
    mask &= ~alleles.str.contains("undetermined", na=False)


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
    assay_units = df['Assay']['Units'].str.replace("\[nM\]", "nM")
    assay_method = df['Assay']['Method/Technique'].str.lower()
    paper = df['Reference']['Title']

    df_clean = pd.DataFrame({})
    df_clean['Epitope'] = epitopes
    df_clean['MHC Allele'] = alleles
    df_clean['MHC Class'] = mhc_class
    df_clean['Assay Method'] = assay_method
    df_clean['Assay Units'] = assay_units
    df_clean['Assay Value'] =  binding_scores
    df_clean['Binder'] = categories
    df_clean['Paper'] = paper

    logging.info("Final DataFrame with %d entries", len(df_clean))

    return df_clean

# looked at intersections of IEDB results, found
# purified MHC radioactivity inhibition assay
# correlates almost perfectly in log-scale with
# cell-bound MHC fluorescence assays.
EQUIV_ASSAY_METHODS = [
    ("purified mhc - radioactivity", "IC50"),
    ("cell bound mhc - fluorescence", "IC50"),
    ("purified mhc - fluorescence", "KD"),
    # ("cell bound mhc - radioactivity", "IC50")
]

def group_by_peptide_and_allele(
        df_peptides,
        max_ic50 = 10**7,
        assay_methods_and_units = EQUIV_ASSAY_METHODS):


    df_peptides['Assay Method'] = df_peptides['Assay Method'].str.lower()
    print "Filtering assay methods"
    assay_mask = np.zeros(len(df_peptides), dtype=bool)
    for (method, units) in assay_methods_and_units:
        mask = df_peptides['Assay Method'].str.contains(method, na=False)
        mask &= df_peptides['Assay Units'].str.contains(units, na=False)
        mask &= df_peptides['Assay Units'].str.contains('nM', na=False)
        print "-- %s : %s nM: %d entries" % (
            method, units, mask.sum()
        )
        assay_mask |= mask
    assay_mask &= ~df_peptides['Assay Value'].isnull()
    assay_mask &= df_peptides['Assay Value'] < max_ic50

    print "%d entries with IC50 (or equivalent) assay values" % \
        assay_mask.sum()

    df_peptides['Assay Mask'] = assay_mask

    records = []

    positive_binding_categories = [
        "Positive",
        "Positive-Low",
        "Positive-Intermediate",
        "Positive-High"
    ]
    print "Generating grouped peptide data"
    for allele, allele_group in df_peptides.groupby(["MHC Allele"]):
        mhc_class = allele_group['MHC Class'].irow(0)
        print "  ", allele, len(allele_group)
        for epitope, group in allele_group.groupby("Epitope"):
            record = OrderedDict()
            record["MHC Allele"] = allele
            record["MHC Class"] = mhc_class
            record["Epitope"] = epitope
            record["Count"] = len(group)

            for label in positive_binding_categories:
                record[label] = (group["Binder"] == label).sum()
            record["Negative"] = (group["Binder"] == "Negative").sum()
            record['Positive-All'] = \
                sum(record[l] for l in positive_binding_categories)
            ic50 = group['Assay Value']
            mask = group['Assay Mask']
            ic50 = ic50[mask]
            record['IC50_Count'] = len(ic50)
            record['IC50_Min'] = ic50.min()
            record['IC50_Max'] = ic50.max()
            record['IC50_Median'] = ic50.median()
            record['IC50_Std'] = ic50.std()
            records.append(record)
    return pd.DataFrame.from_records(records)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    import argparse
    parser = argparse.ArgumentParser(
        description='Download and prepare IEDB peptide binding data')

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
        "--grouped-output",
        default='mhc_grouped.csv')

    args = parser.parse_args()

    df_iedb = load_iedb(args.iedb_filename, args.iedb_url)

    logging.info("# assay results = %d", len(df_iedb))

    print "Epitope lengths"
    print df_iedb['Epitope'].str.len().value_counts()

    if args.iedb_output:
        df_iedb.to_csv(args.iedb_output, index=False)

    df_peptides = group_by_peptide_and_allele(df_iedb)

    print df_peptides
    print "Generated %d allele/peptide pairs" % len(df_peptides)

    if args.grouped_output:
        df_peptides.to_csv(args.grouped_output, index=False)

