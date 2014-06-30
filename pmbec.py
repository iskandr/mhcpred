def read_coefficients(filename = 'pmbec.mat'):

	d = {}
	with open(filename, 'r') as f:
		lines = [line for line in f.read().split('\n') if len(line) > 0]
		header = lines[0]
		print header
		residues = [x for x in header.split(' ') if len(x) == 1 and x != ' ' and x != '\t']
		print residues
		assert len(residues) == 20
		for line in lines[1:]:
			cols = [x for x in line.split(' ') if len(x) > 0 and x != ' ' and x != '\t']
			assert len(cols) == 21, "Expected 20 values + letter, got %s" % cols
			row_letter = cols[0]
			for i, col in enumerate(cols[1:]):
				col_letter = residues[i]
				assert col_letter != ' ' and col_letter != '\t'
				value = float(col)
				key = "%s%s" % (row_letter, col_letter)
				d[key] = value
	return d

if __name__ == '__main__':
	d = read_coefficients()
	print "PMBEC matrix"
	for k in sorted(d):
		print k, d[k]
	