def parse_fasta(s):
    lines = s.split("\n")
    lines = [l for l in lines if len(l) > 0]

    i = 0
    d = {} 
    curr_seq = ""

    # parse FASTA file
    while i < len(lines):
        curr = lines[i]
        i += 1
        if curr.startswith(">"):
            if curr_seq:
                d[k] = curr_seq
            k = curr[1:].split(" ")[0]
            curr_seq = ""
        elif curr:
            curr_seq += curr
    return d