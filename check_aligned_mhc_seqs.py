import sys
from fasta import parse_fasta

filename = sys.argv[1] if len(sys.argv) > 1 else "multispecies_aligned.fasta"

with open(filename,'r') as f:
    d = parse_fasta(f.read())

seqs = d.values()
seqlen = len(seqs[0])
n_seqs = len(seqs)

assert all(len(v)==seqlen for v in seqs)
conserved = set([])
for i in xrange(seqlen):
    count = sum(s[i] != "-" for s in seqs)
    alleles = [allele for allele, seq in d.iteritems() if seq[i] != "-"]
    species = set([allele.split("-")[0] for allele in alleles])
    print i, count, "/", n_seqs, list(sorted(species))
    if count < 25:
        print "---", alleles
    elif count == n_seqs:
        conserved.add(i)
    elif count > n_seqs - 25:
        print "---", [allele for allele, seq in d.iteritems() if seq[i] == "-"]
print "Conserved positions: %d %s" % (len(conserved), list(sorted(conserved)))