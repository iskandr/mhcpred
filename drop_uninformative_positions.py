import argparse
import collections

from parsing import parse_fasta_mhc_files
from seq_helpers import positional_letter_counts


parser = argparse.ArgumentParser(
    description=\
        'Drop positions from a collection of strings which are always the same'
    )

parser.add_argument(
    "--input-file",
    required=True,
    help="Input FASTA file",
)

parser.add_argument(
    "--output-file",
    required=True,
    help="Output FASTA file",
)

parser.add_argument(
    "--min-different-fraction",
    type=float,
    default=0.01,
    help="Fraction of seqs which must be different for position "
         "to be considered 'informative'"
)

if __name__ == "__main__":
    args = parser.parse_args()
    seqs_dict = parse_fasta_mhc_files([args.input_file])
    counts = positional_letter_counts(seqs_dict.values())
    max_count = int( (1.0 - args.min_different_fraction) * len(seqs_dict))
    print "Max same for position to be kept: %d" % max_count
    keep = [i for i,c in enumerate(counts) if c.most_common()[0][1] < max_count]
    print "Keeping %d/%d positions" % (len(keep), len(counts))
    print "--", keep
    with open(args.output_file, 'w') as f:
        for allele in sorted(seqs_dict.keys()):
            seq = seqs_dict[allele]
            subset = "".join([seq[i] for i in keep])
            f.write(">%s\n%s\n" % (allele, subset))