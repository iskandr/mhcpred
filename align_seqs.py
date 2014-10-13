import sys
import argparse
import collections

from parsing import parse_fasta_mhc_dirs, parse_aa_table

import numpy as np


parser = argparse.ArgumentParser(
    description='Align MHC sequences')

parser.add_argument(
    "dir",
    help="Directory which contains MHC sequence fasta files to align",
)
parser.add_argument(
    "--reference-allele",
    default = "HLA-A*02:01",
    help="Allele to align against",
)

parser.add_argument(
    "--output-file",
    type = str,
    help = "Path to output FASTA file",
    default="aligned_output.fa"
)

parser.add_argument(
    "--exclude",
    type = str,
    help = "Exclude alleles which contain this substring",
    default = "",
)

def align(ref, seq):
    """
    Delete positions from seq one at a time until same length as ref
    """
    n = len(ref)

    assert len(seq) >= n, (ref, seq, len(ref), len(seq))
    n_extra = n - len(seq)

    blosum50 = parse_aa_table("BLOSUM50")

    # choose best start position (trimming off large insertions at the
    # beginning of the sequence
    best_score = -1000
    aligned = seq
    for i in xrange(len(seq) - len(ref)):
        score = sum(blosum50[x+y] for (x,y) in zip(seq[i:], ref))
        if score > best_score:
            best_score = score
            aligned = seq[i:]

    while len(aligned) > n:
        # chpoose a position to delete
        best = None
        best_score = -1000
        for i in xrange(len(aligned)):
            candidate = aligned[:i] + aligned[i+1:]
            score = sum(blosum50[x+y] for (x,y) in zip(candidate, ref))
            if score > best_score:
                best = candidate
                best_score = score
        aligned = best
    return aligned


if __name__ == '__main__':
    args = parser.parse_args()
    exclude_alleles = [allele for allele in args.exclude.split(",") if allele]
    seqs = parse_fasta_mhc_dirs(
        [args.dir],
        exclude_alleles=exclude_alleles)
    assert args.reference_allele in seqs, seqs.keys()
    ref = seqs[args.reference_allele]
    print "Reference", ref
    print "Length", len(ref)
    aligned = collections.OrderedDict()
    for allele in sorted(seqs.keys()):
        seq = seqs[allele]
        if len(seq) < len(ref):
            print "Skipping %s (len=%d)" % (allele, len(seq))
        elif any(c in seq for c in ("X", "Z", "B", "J")):
            print "Incomplete sequence for %s" % allele
        else:
            print ">", allele
            aligned_seq = align(ref, seq)
            mismatch = "".join(
                x if x != y else "_" for (x,y) in zip(aligned_seq,ref)
            )
            print mismatch
            aligned[allele] = aligned_seq

    with open(args.output_file, 'w') as f:
        for allele, seq in aligned.iteritems():
            f.write(">%s\n%s\n" % (allele, seq))
            f.flush()