#! /usr/bin/env python

import numpy as np

def kappa(R, max_count=None, check_bin=True, verbosity=0):
    """Calculate the `kappa' for a binary matrix according to algorithm in Kenneth P. Ewing, ``Bounds for the Distance Between Relations'' (2021-05-03), arXiv:NNNN.NNN [math.GT]."""
    assert (not check_bin) or (np.max(R) <= 1), TypeError("Input is not a binary matrix: {0}".format(R))
    assert (not max_count) or (isinstance(max_count, int) and max_count >= 0), AssertionError("Optional max_count is not a natural number: {0}".format(max_count))
    assert (not verbosity) or (isinstance(verbosity, int) and verbosity >= 0), AssertionError("Verbosity is not a natural number: {0}".format(verbosity))

    # build maximal x-groups: O(rows * cols); for speed use sets since mutable hashes
    #
    # initialize with first col
    x_groups = [(set(list(np.nonzero(R[:,0])[0])), set([0]))]
    if verbosity > 1:
        print("R:")
        print(R)
        print("col: {0} xs: {1} x_groups: {2}".format(0, x_groups[0][0], x_groups))
    # loop over other col indices
    for c in range(1, R.shape[1]):
        # get first column's nonzero row indices
        xs = set(list(np.nonzero(R[:,c])[0]))
        # initialize new_xgs with first column
        new_xgs = [(xs, set([c]))]
        # initialize disjoint flag
        # dj_flag = True
        # loop over indices of x_groups
        for i in range(len(x_groups)):
            # if new_xgs[0] overlaps,...
            if not x_groups[i][0].isdisjoint(new_xgs[0][0]):
                if verbosity > 2:
                    print(" overlap> new_xgs[0]: {0} x_groups[{1}]: {2}"
                          .format(new_xgs[0][0], i, x_groups[i][0]))
                # ...unset the disjoint flag
                # dj_flag = False
                # ...expand new_xgs[0] with x-groups[i]
                new_xgs[0] = (new_xgs[0][0].union(x_groups[i][0]),
                              new_xgs[0][1].union(x_groups[i][1]))
            # else add x_groups[i] to new_xgs
            else:
                if verbosity > 2:
                    print(" disjoint> new_xgs[0]: {0} x_groups[{1}]: {2}"
                          .format(new_xgs[0][0], i, x_groups[i][0]))
                new_xgs = new_xgs + [x_groups[i]]
            if verbosity > 2:
                print(" new> new_xgs: {0} x_groups: {1}"
                      .format(new_xgs, x_groups))
        # update x_groups to new_xgs
        x_groups = new_xgs
        if verbosity > 1:
            print("col: {0} xs: {1} x_groups: {2}"
                  .format(c, xs, x_groups))
        # if was disjoint...
        # if dj_flag is True:
            # ...start new x_group
            # x_groups = x_groups + [probe]
            # if verbosity > 2:
                # print(" disjoint: {0} and {1}".format(probe, c))
    if verbosity > 0:
        print("x_groups: {0}".format(x_groups))

    # blockcounts is sorted length of each x_group's columns: O(rows)
    blockcounts = sorted([len(xg[1]) for xg in x_groups])

    # blocksums is cumulative sum of sorted blockcounts: O(rows)
    blocksums = np.cumsum(blockcounts)

    # now calculate the kappa: O(rows)
    cap = max_count or R.shape[1]
    m = np.argmax(blocksums[blocksums <= cap])
    if verbosity > 0:
        print("blockcounts: {0}".format(blockcounts))
        print("blocksums: {0}".format(blocksums))
        print("cap: {0}".format(cap))
        print("m: {0}".format(m))
    if len(blocksums) == 1:
        return 0
    elif cap >= R.shape[1]:
        return blocksums[m-1]
    elif blocksums[m] + blockcounts[m] > cap:
        return blocksums[m-1]
    else:
        return blocksums[m]


def rel_diff(R1, R2, check_bin=True):
    """One-for-one remove from relation `R1' any columns found in `R2', both represented as binary matrices"""
    assert (not check_bin) or np.all([(np.max(R) <= 1) for R in [R1, R2]]), AssertionError("Inputs not binary matrices: {0} {1}".format(R1, R2))
    assert R1.shape[1] == R2.shape[1], AssertionError("Inputs don't have same number of rows: {0} {1}".format(R1, R2))

    # initialize with no agreeing columns: O(max(cols))
    R1_agrs = []
    R2_agrs = []
    R1_disagrs = list(range(R1.shape[1]))

    # check all source columns: O(cols1 * cols2 * rows)
    for i in range(R1.shape[1]):
        # check all target columns
        for j in range(R2.shape[1]):
            # skip agreeing target; try next target
            if j in R2_agrs:
                continue
            # if agree with new target update lists; try next source
            if not np.any(R1[:,i] - R2[:,j]):       # presumably O(rows)
                R1_agrs = R1_agrs + [i]
                R2_agrs = R2_agrs + [j]
                R1_disagrs.remove(i)
                break
    return np.take(R1, R1_disagrs, axis=1)

def rel_dist_bound(R1, R2, check_bin=True, verbosity=0):
    """Calculate bound on distance between two relations using `kappa'"""
    assert (not check_bin) or np.all([(np.max(R) <= 1) for R in [R1, R2]]), AssertionError("Inputs not binary matrices: {0} {1}".format(R1, R2))
    assert R1.shape[1] == R2.shape[1], AssertionError("Inputs don't have same number of rows: {0} {1}".format(R1, R2))

    # calculate disagreeing columns in each direction
    R1_disagrs = rel_diff(R1, R2, check_bin=False)
    R2_disagrs = rel_diff(R2, R1, check_bin=False)
    if verbosity > 0:
        print("R1 disagreeing: \n{0}".format(R1_disagrs))
        print("R2 disagreeing: \n{0}".format(R2_disagrs))

    # calculate the kappas
    k12 = kappa(R2_disagrs, R1_disagrs.shape[1], verbosity=verbosity)
    k21 = kappa(R1_disagrs, R2_disagrs.shape[1], verbosity=verbosity)
    if verbosity > 0:
        print("kappa(R1,R2): {0}".format(k12))
        print("kappa(R2,R1): {0}".format(k21))
        print()
        print("dist_bound(R1, R2): max({0}, {1}) - min({2} + {3}, {4} + {5})"
              .format(R1.shape[1], R2.shape[1], R1.shape[1] - R1_disagrs.shape[1], k12,
                R2.shape[1] - R2_disagrs.shape[1], k21))

    return max(R1.shape[1], R2.shape[1]) - min(R1.shape[1] - R1_disagrs.shape[1] + k12, R2.shape[1] - R2_disagrs.shape[1] + k21)

