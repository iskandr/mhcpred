

def read_coefficients(filename = 'pmbec.mat', key_type='row', verbose=True):
	"""
	Parameters
	------------

	filename : str
		Location of PMBEC coefficient matrix

	key_type : str
		'row' : every key is a single amino acid,
				which maps to a dictionary for that row
		'pair' : every key is a tuple of amino acids
		'pair_string' : every key is a string of two amino acid characters

	verbose : bool
		Print rows of matrix as we read them
	"""
	d = {}
	if key_type == 'row':
		def add_pair(row_letter, col_letter, value):
			if row_letter not in d:
				d[row_letter] = {}
			d[row_letter][col_letter] = value
	elif key_type == 'pair':
		def add_pair(row_letter, col_letter):
			d[(row_letter, col_letter)] = value
	else:
		assert key_type == 'pair_string', \
			"Invalid dictionary key type: %s" % key_type
		def add_pair(row_letter, col_letter, value):
			d["%s%s" % (row_letter, col_letter)] = value

	with open(filename, 'r') as f:
		lines = [line for line in f.read().split('\n') if len(line) > 0]
		header = lines[0]
		if verbose:
			print header
		residues = [x for x in header.split(' ') if len(x) == 1 and x != ' ' and x != '\t']
		assert len(residues) == 20
		if verbose:
			print residues
		for line in lines[1:]:
			cols = [
				x
				for x in line.split(' ')
				if len(x) > 0 and x != ' ' and x != '\t'
			]
			assert len(cols) == 21, "Expected 20 values + letter, got %s" % cols
			row_letter = cols[0]
			for i, col in enumerate(cols[1:]):
				col_letter = residues[i]
				assert col_letter != ' ' and col_letter != '\t'
				value = float(col)
				add_pair(row_letter, col_letter, value)

	return d

if __name__ == '__main__':
	d = read_coefficients(key_type='pair_string')
	print "PMBEC matrix"
	for k in sorted(d):
		print k, d[k]
