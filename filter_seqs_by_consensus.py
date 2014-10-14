import argparse
import collections

from parsing import parse_fasta_mhc_files
from seq_helpers import positional_letter_counts, most_common_characters

parser = argparse.ArgumentParser(
    description=
        'Filter sequences in a FASTA file by dropping those that mostly'
        'mismatch the consensus'
    )

parser.add_argument(
    "--input-file",
    required=True,
    help="Input FASTA file",
)

parser.add_argument(
    "--output-file",
    required=True,
    help="Output FASTA file"
)

parser.add_argument(
    "--fraction-match-required",
    default=0.5,
    type=float,
)


if __name__ == "__main__":
    args = parser.parse_args()
    seqs_dict = parse_fasta_mhc_files([args.input_file])
    seqs = seqs_dict.values()

    # ensure all sequences are of correct length
    n, consensus = most_common_characters(seqs)


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

    with open(args.output_file, 'w') as f:
        for allele in sorted(filtered_seqs.keys()):
            seq = filtered_seqs[allele]
            f.write(">%s\n%s\n" % (allele, seq))
    print "---"
    print "CONSENSUS"
    print "".join(consensus)

