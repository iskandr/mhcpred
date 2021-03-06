import pandas as pd 
import numpy as np 
import seaborn
import scipy
CUTOFF = 50000
if __name__ == "__main__":

    df = pd.read_csv('mhc.csv')    
    method_col = df['Assay Method'].str.lower()

    dfs = {}

    for assay in ['cell bound', 'purified']:
        assay_mask = method_col.str.contains(assay, na=False)
        for tag in ['fluorescence', 'radio']:
            tag_mask = method_col.str.contains(tag, na=False)
            for units in ['IC50', 'EC50', 'KD']:
                units_mask = df['Assay Units'].str.contains(units, na=False)
                mask = tag_mask & assay_mask & units_mask
                n_entries = mask.sum()
                g = df[mask].groupby(['MHC Allele', 'Epitope'])
                value = g['Assay Value'].median()
                n_epitopes = len(value)
                print "%s/%s/%s: %d entries, %d unique" % (
                    assay, tag, units, n_entries, n_epitopes
                )
                dfs[(assay, tag, units)] = value
    print 
    print
    seen = set([]) 
    for (k1,s1) in dfs.iteritems():
        lsuffix =  "-".join(k1)
        lkey = "Log Assay Value " + lsuffix 
        df1 = s1.reset_index()
	# get rid of lower bound assay values
	bad_mask1 = (df1['Assay Value'] % 5000) == 0
        df1 = df1[~bad_mask1]
        df1['Log Assay Value'] = np.log(df1['Assay Value']) / np.log(CUTOFF)
        for (k2, s2) in dfs.iteritems():
            if k1 != k2 and frozenset([k1, k2]) not in seen:
                seen.add(frozenset([k1, k2]))
                rsuffix =  "-".join(k2)
                rkey = "Log Assay Value " + rsuffix
                df2 = s2.reset_index()
		bad_mask2 = (df2['Assay Value'] % 5000) == 0
		df2 = df2[~bad_mask2]
                df2['Log Assay Value']  = np.log(df2['Assay Value']) / np.log(CUTOFF)
                joined = pd.merge(
                    df1, 
                    df2, 
                    left_on = ('MHC Allele', 'Epitope'),
                    right_on = ('MHC Allele', 'Epitope'),
                    suffixes = (" " + lsuffix, " " + rsuffix), 
                )

                n = len(joined)
                   
                if n > 0:
	            print joined[['MHC Allele', 'Epitope', 'Assay Value ' + lsuffix, 'Assay Value ' + rsuffix]] 
                    print "Intersection of %s and %s has %d entries" % (
                        lsuffix, rsuffix, n
                    )
                    l = joined[lkey]
                    r = joined[rkey]
                    diff = (50000 ** l) - (50000 ** r)
                    abs_diff = np.abs(diff)
                    print "-- Abs Diff: 5th %f, 95th %f,  Median %f"  %(
                        scipy.percentile(abs_diff, 5), 
                        scipy.percentile(abs_diff, 95), 
                        np.median(abs_diff)
                    )
                    if n >= 25:
                        seaborn.jointplot(
                            lkey, 
                            rkey,
                            data = joined,
                            kind = 'kde',
                            xlim = (-0.1, 1.1),
                            ylim = (-0.1, 1.1),
                        )


    seaborn.plt.show()
     
