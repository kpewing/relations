# relations
Python code for topological analysis of (mathematical) relations.

## Dependencies

- `numpy`

## kappa.py

Topological distance between two mathematical relations, represented by binary matrices showing which objects (columns) exhibit which features (rows), can be defined in terms of the number of changes required to transform one matrix into the other and vice versa, somewhat like the [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) for strings (see Robinson, [link to come](). Calculating the exact distance requires considering all possible transformations, which can be exponential in time: $\mathcal{O}(n^{m})$ for relations of lengths *n* and *m*. An upper bound on the distance can, however, be calculated in $\mathcal{O}(n\times m)$ time using the *kappa* value for the disagreeing columns in each of two relations when compared to the other (see [Bounds for the Distance Between Relations]()). 

The code in `kappa.py` exposes the following functions:

kappa.**kappa**(*R*, *max_count*=None, *check_bin*=True, *verbosity*=0)

- Calculate *kappa* for a binary matrix (`numpy.array`) *R* for up to *max_count* columns. If *check_bin* is `True` then confirm that the matrix is binary. Use *verbosity* to see intermediate calculations.

kappa.**rel_dist_bound**(*R1*, *R2*, *check_bin*=True, *verbosity*=0):

- Calculate an upper bound of the relation distance between two relations *R1* an *R2* (`numpy.array`) using the *kappa* algorithm.  If *check_bin* is `True` then confirm that the matrix is binary. Use *verbosity* to see intermediate calculations.

kappa.**rel_diff**(*R1*, *R2*, *check_bin*=True):

- Calculate the difference of two binary matrices *R1* and *R2* (`numpy.array`): returns a new binary matrix after one-for-one removing from *R1* each column that is also present in *R2*. If *check_bin* is `True` then confirm that the matrix is binary.

kappa.**rel_from_dict**(*d*, *default*=0):

- Convert a `python` dict (rows) of dict (cols) representing a relation to a two-dimensional `numpy` array. Missing entries are assigned the *default* value.
- For a sparse array, ensure all rows and all cols appear at least once, and set default.

Command line usage:

```
usage: kappa.py [-h] [--def1 DEF1] [--def2 DEF2] [-m MAX_COUNT] [-v {0,1,2}]
                [-c] [-d]
                rel1 [rel2]

Calculates `kappa' of one relation or the `relation distance' between two
relations according to the `kappa' algorithm in Kenneth P. Ewing, ``Bounds for
the Distance Between Relations'', arXiv:NNNN.NNN. For sparse arrays, ensure
all rows and all cols appear at least once and set defaults.

positional arguments:
  rel1                  Filepath for first relation (JSON).
  rel2                  Filepath for second relation (JSON).

optional arguments:
  -h, --help            show this help message and exit
  --def1 DEF1           Default for sparse first relation.
  --def2 DEF2           Default for sparse second relation.
  -m MAX_COUNT, --max_count MAX_COUNT
                        Number of columns for kappa to map.
  -c, --check_bin       Check that relations are binary matrices.
  -d, --display         Display relation matrices before printing result.
  -v {0,1,2}, --verbosity {0,1,2}
                        Print up to 3 levels of intermediate results of calculations.
```
