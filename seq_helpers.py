import collections

def all_same_length(seqs):
    """
    Ensure all strings are of same length and
    determine what that length is
    """
    seq = seqs[0]
    n = len(seq)
    assert all(len(s) == n for s in seqs)
    return n


def positional_letter_counts(seqs, n=None):
    """
    Given a collection of strings of length `n`,
    return a dictionary of letter counts for each position
    """
    if n is None:
        n = all_same_length(seqs)

    counters = []
    for i in xrange(n):
        c = collections.Counter()
        for s in seqs:
            c[s[i]] += 1
        counters.append(c)
        print i, c
    return counters



def most_common_characters(seqs):
    """
    Given a collection of strings of the same length,
    return the most common character at each position
    """
    n = all_same_length(seqs)
    counters = positional_letter_counts(seqs, n)
    consensus = [c.most_common()[0][0]  for c in counters]
    return n, consensus