
strings = []
alleles = []
with open('MHC_aa_seqs.txt') as f:
	text = f.read()
	for line in text.split('\n'):
		parts = [x for x in line.split(' ') if len(x) > 0]
		assert len(parts) <= 2
		if len(parts) == 2:
			allele, seq = parts
			if allele.startswith('HLA'):
				strings.append(seq)
				alleles.append(allele)

first = strings[0]
keep_position = [True] * len(first)
for i, c in enumerate(first):
	count = 0
	total = 0
	for s in strings:
		if s[i] == c:
			count += 1
		total += 1
	fraction =  float(count)/total
	print i, c, count, "/", total, fraction 
	if fraction > 0.99:
		print "-- Skipping position %d" % i 
		keep_position[i] = False 

import numpy as np 
print "# skipped = %d" % (len(keep_position) - np.sum(keep_position))

