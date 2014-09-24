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
from immuno.common import find_paths

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
    paths = find_paths(
        directory_string = args.input_dir,
        extensions = [".fa", ".fasta"])
    assert len(paths) > 0, "No files found"


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
                if args.exclude and args.exclude in allele:
                    print "Skipping excluded allele", allele 
                    continue 

                allele = allele.replace("_", "*")
                
                seq = str(record.seq)
                if len(seq) < args.min_length:
                    print "Skipping", allele, "length =", len(seq)
                    continue 

                allele = ":".join(allele.split(":")[:2])
                if allele in seq:
                    # expect all 4-digit alleles to correspond to the same
                    # protein sequence
                    assert seqs[allele] == seq, (record, seq)
                else:
                    seqs[allele] = seq
    if args.output_file:
        with open(args.output_file, 'w') as f:
            for allele in sorted(seqs.keys()):
                seq = seqs[allele]
                f.write(">%s %d bp\n%s\n" % (allele, len(seq), seq))
    else:
        for k,v in sorted(seqs.iteritems()):
            print ">",k
            print v 
    