import argparse
import collections

from parsing import parse_fasta_mhc_files


parser = argparse.ArgumentParser(
    description='Drop positions which are always identical in a collection'
    'of strings, write output to a NetMHCpan pseudosequence file'
    )

parser.add_argument(
    "file",
    help="Directory which contains MHC sequence fasta files to align",
)

parser.add_argument(
    "--fraction-match-required",
    default=0.1,
    type=float,
)


def positional_letter_counts(seqs, n):
    counters = []
    for i in xrange(n):
        c = collections.Counter()
        for s in seqs:
            c[s[i]] += 1
        counters.append(c)
        print i, c
    return counters


if __name__ == "__main__":
    args = parser.parse_args()
    seqs_dict = parse_fasta_mhc_files([args.file])
    seqs = seqs_dict.values()

    # ensure all sequences are of correct length
    seq = seqs[0]
    n = len(seq)
    assert all(len(s) == n for s in seqs)

    counters = positional_letter_counts(seqs, n)
    consensus = [c.most_common()[0][0]  for c in counters]
    print "CONSENSUS"
    print "".join(consensus)

    # drop sequences which match less than 10% of the consensus residues
    n_dropped = 0
    filtered_seqs = {}
    for allele in sorted(seqs_dict.keys()):
        seq = seqs_dict[allele]
        n_match = sum(x == y for (x,y) in zip(seq, consensus))
        if n_match < args.fraction_match_required * n:
            print "Dropping sequence %s: %s" % (allele, seq)
            n_dropped += 1
        else:
            filtered_seqs[allele] = seq
    print "Total # dropped: %d" % n_dropped

