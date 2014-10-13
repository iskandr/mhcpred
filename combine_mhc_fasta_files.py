#!/usr/bin/env python2

# Copyright (c) 2014. Mount Sinai School of Medicine
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from Bio.SeqIO.FastaIO import FastaIterator
import pandas as pd
from parsing import parse_fasta_mhc_dirs
parser = argparse.ArgumentParser()

parser.add_argument(
    "--input-dir",
    required = True,
    type = str,
    help = "Path to FASTA files containing MHC protein sequences"
)

parser.add_argument(
    "--output-file",
    type = str,
    help = "Path to output FASTA file"
)

parser.add_argument(
    "--min-length",
    type = int,
    help = "Minimum sequence length",
    default = 181,
)

parser.add_argument(
    "--exclude",
    type = str,
    help = "Exclude alleles which contain this substring",
    default = "",
)



if __name__ == '__main__':
    args = parser.parse_args()
    exclude_alleles = args.exclude.split(",")
    seqs = parse_fasta_mhc_dirs(
        [args.dir],
        min_length = args.min_length,
        exclude_alleles=exclude_alleles)

    assert len(seqs) > 0, "No sequences found in directory %s" % args.dir
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for allele in sorted(seqs.keys()):
                seq = seqs[allele]
                f.write(">%s %d bp\n%s\n" % (allele, len(seq), seq))
    else:
        for k,v in sorted(seqs.iteritems()):
            print ">",k
            print v
