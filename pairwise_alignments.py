import sys 
import argparse

import nwalign as nw 

from fasta import parse_fasta



parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-filename",
    type = str, 
    help = "Path to FASTA files containing MHC protein sequences"
)

parser.add_argument(
    "--output-filename",
    type = str, 
    help = "Path to output FASTA file"
)

parser.add_argument(
    "--reference-allele",
    default = "HLA-B*08:01",
    type = str, 
    help = "Which allele to align against"
)


if __name__ == '__main__':
    args = parser.parse_args()
    print args

    with open(args.input_filename,'r') as f:
        d = parse_fasta(f.read())

    allele = args.reference_allele

    if allele in d:
        refseq = d[allele]
    else:
        refseq = d[allele.replace("*", "")]

    result = {}
    for k,v in d.iteritems():
        x, y = nw.global_align(refseq, v, gap_open=-40, gap_extend=-20, matrix='BLOSUM50')
        good_positions = [i for i,xi in enumerate(x) if xi != "-"]
        x_subset = "".join(x[i] for i in good_positions)
        y_subset = "".join(y[i] for i in good_positions)
        result[k] = y_subset
    if args.output_filename:
        with open(args.output_filename, 'w') as f:
            for k,v in result.iteritems():
                f.write(">%s\n%s\n" % (k, v))
