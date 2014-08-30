
AMINO_ACID_LETTERS =list(sorted([
    'G', 'P',
    'A', 'V',
    'L', 'I',
    'M', 'C',
    'F', 'Y', 
    'W', 'H', 
    'K', 'R',
    'Q', 'N', 
    'E', 'D',
    'S', 'T',
]))

AMINO_ACID_PAIRS = ["%s%s" % (x,y) for y in AMINO_ACID_LETTERS for x in AMINO_ACID_LETTERS]

AMINO_ACID_PAIR_POSITIONS = dict( (y, x) for x, y in enumerate(AMINO_ACID_PAIRS))