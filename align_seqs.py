import sys
import argparse
import collections
from itertools import izip

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
    default="HLA-C*08:38",
    help="Allele to align against",
)

parser.add_argument(
    "--output-file",
    type=str,
    help="Path to output FASTA file",
    default="aligned_output.fa"
)

parser.add_argument(
    "--require-substring",
    type=str,
    default="",
    help="Allele name must contain this substring"
)

parser.add_argument(
    "--exclude",
    type=str,
    help="Exclude alleles which contain this substring",
    default="",
)

parser.add_argument(
    "--coeff-file",
    type=str,
    default="BLOSUM50",
    help="File containing BLOSUM matrix coefficients"
)

class Aligner(object):

    def __init__(
            self,
            coefficient_matrix_filename="BLOSUM50",
            exact_match_bonus = 0):
        self.coeffs = parse_aa_table(coefficient_matrix_filename)
        self.nested_coeffs_dict = {}
        for k,v in self.coeffs.iteritems():
            x, y = k
            if x not in self.nested_coeffs_dict:
                self.nested_coeffs_dict[x] = {}
            if x == y:
                v += exact_match_bonus
            self.nested_coeffs_dict[x][y] = v

    def align(self, ref, seq, maxdels_per_iter=50):
        """
        Delete positions from seq one at a time until same length as ref
        """
        n = len(ref)

        assert len(seq) >= n, (ref, seq, len(ref), len(seq))
        n_extra = n - len(seq)

        coeffs = self.nested_coeffs_dict

        # choose best start position (trimming off large insertions at the
        # beginning of the sequence
        best_score = 0
        aligned = seq
        print "ref", ref
        print "seq", aligned
        for i in xrange(len(seq) - len(ref) + 1):
            candidate = seq[i:]
            n_matches = sum(x==y for (x,y) in izip(candidate, ref))
            print i, n_matches
            if n_matches > best_score:
                best_score = n_matches
                aligned = candidate

        # after a hopefully faster first phase of getting rid of a
        # non-matching prefix, follow up with approximate-match iterative
        # deletions until same length as reference
        while len(aligned) > n:
            # choose a range of positions to delete
            best = None
            best_score = -np.inf
            best_score_len = 0
            best_i = 0
            best_j = 0
            for i in xrange(len(aligned)):
                maxdels = min(len(aligned) - len(ref), maxdels_per_iter)
                for j in xrange(i+1, min(len(aligned), i+maxdels+1)):
                    candidate = aligned[:i] + aligned[j:]
                    score = sum(coeffs[x][y] for (x,y) in izip(candidate, ref))
                    delsize = j - i
                    if score > best_score:
                        best = candidate
                        best_score = score
                        best_score_len = delsize
                    elif score == best_score and delsize  > best_score_len:
                        best = candidate
                        best_score = score
                        best_score_len = delsize
            aligned = best

        return aligned


if __name__ == '__main__':
    args = parser.parse_args()
    exclude_alleles = [allele for allele in args.exclude.split(",") if allele]
    require_substrings = [
        substr
        for substr in args.require_substring.split(",")
        if substr]
    seqs = parse_fasta_mhc_dirs(
        [args.dir],
        exclude_allele_substrings=exclude_alleles,
        require_allele_substrings=require_substrings)
    assert args.reference_allele in seqs, seqs.keys()
    ref = seqs[args.reference_allele]

    print "Reference", ref
    print "Length", len(ref)

    aligned = collections.OrderedDict()
    aligner = Aligner(args.coeff_file)

    for allele in sorted(seqs.keys()):
        seq = seqs[allele]
        if allele != "Mane-A4*01:03":
            continue
        if len(seq) < len(ref):
            print "Skipping %s (len=%d)" % (allele, len(seq))
        elif any(c in seq for c in ("X", "Z", "B", "J")):
            print "Incomplete sequence for %s" % allele
        else:
            print ">", allele
            aligned_seq = aligner.align(ref, seq)
            mismatch = "".join(
                x if x != y else "_" for (x,y) in zip(aligned_seq,ref)
            )
            print mismatch
            assert len(aligned_seq) == len(ref)
            aligned[allele] = aligned_seq

    with open(args.output_file, 'w') as f:
        for allele, seq in aligned.iteritems():
            f.write(">%s\n%s\n" % (allele, seq))
            f.flush()