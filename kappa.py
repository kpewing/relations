#! /usr/bin/env python

# Copyright 2021 Kenneth P. Ewing
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def kappa(R, max_count=None, check_bin=True, verbosity=0):
    """Calculate the `kappa' for a binary matrix according to algorithm in Kenneth P. Ewing, ``Bounds for the Distance Between Relations'', arXiv:2105.01690."""
    assert (not check_bin) or (np.size(R) == 0) or (np.max(R) <= 1), TypeError("Input is not a binary matrix: {0}".format(R))
    assert (not max_count) or (isinstance(max_count, int) and max_count >= 0), TypeError("Optional max_count is not a natural number: {0}".format(max_count))
    assert (not verbosity) or (isinstance(verbosity, int) and verbosity >= 0), TypeError("Verbosity is not a natural number: {0}".format(verbosity))

    # empty relation has kappa = 0
    #
    if np.size(R) == 0:
        return 0

    # build maximal x-groups: O(rows * cols); for speed use sets since mutable hashes
    #
    # initialize with first col
    x_groups = [(set(list(np.nonzero(R[:,0])[0])), set([0]))]
    if verbosity > 1:
        print("R:")
        print(R)
        print("col: 0")
        print(" xs: {0}".format(x_groups[0][0]))
        print(" x_groups: {0}".format(x_groups))
        print()
    # loop over other col indices
    for c in range(1, R.shape[1]):
        # get first column's nonzero row indices
        xs = set(list(np.nonzero(R[:,c])[0]))
        # initialize new_xgs with first column
        new_xgs = [(xs, set([c]))]
        # loop over indices of x_groups
        for i in range(len(x_groups)):
            # if new_xgs[0] overlaps,...
            if not x_groups[i][0].isdisjoint(new_xgs[0][0]):
                if verbosity > 2:
                    print(" overlap> new_xgs[0]: {0} x_groups[{1}]: {2}"
                          .format(new_xgs[0][0], i, x_groups[i][0]))
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
            print("col: {0}".format(c))
            print(" xs: {0}".format(xs))
            print(" x_groups: {0}".format(x_groups))
            print()
    if verbosity == 1:
        print("x_groups: {0}".format(x_groups))
        print()

    # blockcounts is sorted length of each x_group's columns: O(rows)
    blockcounts = sorted([len(xg[1]) for xg in x_groups])

    # blocksums is cumulative sum of sorted blockcounts: O(rows)
    blocksums = np.cumsum(blockcounts)

    # now calculate the kappa: O(rows)
    cap = max_count or R.shape[1]
    if np.any(blocksums <= cap):
        m = np.argmax(blocksums[blocksums <= cap])
    else:
        m = 0
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


def rel_dist_bound(R1, R2, check_bin=True, verbosity=0):
    """Calculate upper bound of Michael Robinson's `relation distance' between two relations using `kappa' algorithm."""
    assert (not check_bin) or np.all([(np.max(R) <= 1) for R in [R1, R2]]), AssertionError("Inputs not binary matrices: {0} {1}".format(R1, R2))
    assert R1.shape[0] == R2.shape[0], AssertionError("Inputs don't have same number of rows: {0} {1}".format(R1, R2))

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


def rel_diff(R1, R2, check_bin=True):
    """One-for-one remove from relation `R1' any columns found in `R2', both represented as binary matrices"""
    assert (not check_bin) or np.all([(np.max(R) <= 1) for R in [R1, R2]]), AssertionError("Inputs not binary matrices: {0} {1}".format(R1, R2))
    assert R1.shape[0] == R2.shape[0], AssertionError("Inputs don't have same number of rows: {0} {1}".format(R1, R2))

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


def rel_from_dict(d, default=0):
    """Convert a python dict (rows) of dict (cols) representing a relation to a two-dimensional `numpy' array. Missing entries are assigned the default value.

    For sparse array, ensure all rows and all cols appear at least once, and set default."""
    assert isinstance(d, dict), TypeError("Input must be a dict of dict representing a relation:{0}".format(d))
    assert isinstance(default, int), TypeError("Default must be an integer: {0}".format(default))

    r_count = len(d)
    rows = {z[0]:z[1] for z in zip(list(d), range(r_count))}
    cols = {}
    ar = []
    for r in rows:
        assert isinstance(d[r], dict), TypeError("Input must be a dict of dict representing a relation:{0}".format(d[r]))
        for c in d[r]:
            if c not in cols:
                cols[c] = len(cols)
                ar = ar + [[default] * r_count]
            ar[cols[c]][rows[r]] = int(d[r][c])
    return np.array(ar).T


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="""Calculates `kappa' of one relation or the `relation distance' between
        two relations according to the `kappa' algorithm in Kenneth P. Ewing, ``Bounds for the
        Distance Between Relations'' arXiv:NNNN.NNN. For sparse arrays, ensure
        all rows and all cols appear at least once and set defaults.
        """)
    group = parser.add_mutually_exclusive_group()
    parser.add_argument('rel1', nargs=1, default=None, help='Filepath for first relation (JSON).')
    parser.add_argument('--def1', type=int, default=0, help='Default for sparse first relation.')
    group.add_argument('rel2', nargs='?', default=None, help='Filepath for second relation (JSON).')
    group.add_argument('--def2', type=int, default=0, help='Default for sparse second relation.')
    group.add_argument('-m', '--max_count', type=int, default=0, help='Number of columns for kappa to map.')
    parser.add_argument('-c', '--check_bin', action='store_true', help='Check that relations are binary matrices.')
    parser.add_argument('-d', '--display', action='store_true', help='Display relation matrices before printing result.')
    parser.add_argument('-v', '--verbosity', type=int, choices=range(3), default=0, help='Print up to 3 levels of intermediate results of calculations.')
    args = parser.parse_args()

    if args.rel1:
        with open(args.rel1[0], 'r') as fd:
            # r1 = pd.read_json(fd).to_numpy()
            r1 = rel_from_dict(json.load(fd), default=args.def1)
        if args.display is True:
            np.set_printoptions(threshold=r1.shape[0] * r1.shape[1])
            print(r1); print()
    if args.rel2:
        with open(args.rel2[0], 'r') as fd:
            # r2 = pd.read_json(fd).to_numpy()
            r2 = rel_from_dict(json.load(fd), default==args.def2)
        if args.display is True:
            np.set_printoptions(threshold=r2.shape[0] * r2.shape[1])
            print(r2); print()
        print(rel_dist_bound(r1, r2, check_bin=args.check_bin, verbosity=args.verbosity))
    else:
        if args.max_count == 0:
            mc = None
        else:
            mc = args.max_count
        print(kappa(r1, max_count=mc, check_bin=args.check_bin, verbosity=args.verbosity))
