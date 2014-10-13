from immuno.common import find_paths
from Bio.SeqIO.FastaIO import FastaIterator
def parse_fasta_mhc_dirs(
        dirs,
        min_length=0,
        exclude_alleles=None):
    paths = []
    for d in dirs:
        paths.extend(
            find_paths(
                directory_string = d,
                extensions = [".fa", ".fasta"]))
    return parse_fasta_mhc_files(
            paths,
            min_length=min_length,
            exclude_alleles=exclude_alleles)

def parse_fasta_mhc_files(paths, min_length=0, exclude_alleles=[]):
    seqs = {}
    for path in paths:
        with open(path, 'r') as f:
            for record in FastaIterator(f):
                desc = record.description
                fields = desc.split(" ")
                if len(fields) ==1:
                    allele = fields[0]
                else:
                    allele = fields[1]
                    if "-" not in allele and fields[0].startswith("HLA"):
                        allele = "HLA-" + allele
                if allele.endswith("N") or allele.endswith("Q"):
                    continue
                if exclude_alleles and any(
                        exclude in allele for exclude in exclude_alleles):
                    print "Skipping excluded allele", allele
                    continue

                allele = allele.replace("_", "*")

                seq = str(record.seq)
                if len(seq) < min_length:
                    print "Skipping", allele, "length =", len(seq)
                    continue

                allele = ":".join(allele.split(":")[:2])
                if allele in seq:
                    # expect all 4-digit alleles to correspond to the same
                    # protein sequence
                    assert seqs[allele] == seq, (record, seq)
                else:
                    seqs[allele] = seq
    return seqs

def parse_aa_table(filename, coeff_type = int):
    """
    Parse a table of pairwise amino acid coefficient (e.g. BLOSUM50)
    """
    with open(filename, 'r') as f:
        contents = f.read()
        lines = contents.split("\n")
        # drop comments
        lines = [line for line in lines if not line.startswith("#")]
        # drop CR endline characters
        lines = [line.replace("\r", "") for line in lines]
        labels = lines[0].split()
        assert len(labels) == 20, \
            "Expected 20 amino acids but first line '%s' has %d fields" % (
            lines[0],
            len(labels)
            )
        coeffs = {}
        for line in lines[1:]:
            fields = line.split()
            assert len(fields) == 21, \
                "Expected letter + 20 coefficients but '%s 'has %d fields" % (
                    line, len(fields)
                )
            x = fields[0]
            for i, coeff_str in enumerate(fields[1:]):
                y = labels[i]
                coeff = coeff_type(coeff_str)
                coeffs[x+y] = coeff
        return coeffs
