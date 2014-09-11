from collections import Counter 

if __name__ == '__main__':

	strings = []
	alleles = []
	with open('netmhcpan_pseudo_mhc.txt', 'r') as f:
		text = f.read()
		for line in text.split('\n'):
			parts = [x for x in line.split(' ') if len(x) > 0]
			assert len(parts) <= 2
			if len(parts) == 2:
				allele, seq = parts
				strings.append(seq)
				alleles.append(allele)

	first = strings[0]
	keep_position = [True] * len(first)
	
	for i in xrange(len(first)):
		counter = Counter()
		for s in strings:
			counter[s[i]] += 1

		total = sum(counter.values())
		most_common_key, most_common_count = counter.most_common()[0]
		fraction =  float(most_common_count)/total
		print i, most_common_key, most_common_count, "/", total, fraction 
		if fraction > 0.995:
			print "-- Skipping position %d" % i 
			keep_position[i] = False 


	import numpy as np 
	print "# skipped = %d" % (len(keep_position) - np.sum(keep_position))

	with open('MHC_aa_seqs.csv', 'w') as f:
		f.write("Allele,Residues\n")
		for i, (seq, allele) in enumerate(zip(strings,alleles)):
			subseq = "".join([c for i, c in enumerate(seq) if keep_position[i]])
			f.write("%s,%s\n" % (allele, subseq))

