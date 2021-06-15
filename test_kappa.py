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

import unittest
from kappa import *

class TestKappa(unittest.TestCase):

    def setUp(self):
        self.R = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0, 0], \
                           [0, 0, 1, 1, 0, 0, 0, 0, 0, 0], \
                           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0], \
                           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
        self.R4 = np.array([[1, 0, 1, 1, 0, 0, 1, 1, 1, 1], \
                            [0, 1, 0, 1, 0, 0, 0, 1, 0, 1], \
                            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0], \
                            [1, 1, 1, 0, 1, 0, 0, 0, 0, 1], \
                            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=np.uint8)
        return

    def test_example4_8(self):
        self.assertEqual(kappa(self.R[:,1:4], max_count=5),0)
        return

    def test_example4_9(self):
        self.assertEqual(kappa(self.R[:,0:4], max_count=5),1)
        return

    def test_example4_10(self):
        self.assertEqual(kappa(self.R, max_count=4),1)
        return

    def test_example4_11(self):
        self.assertEqual(kappa(self.R, max_count=5),3)
        return

    def test_example4_17(self):
        R1 = np.array([[1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
                       [0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
                       [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                       [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
                       [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=np.uint8)
        R2 = np.array([[0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                       [0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
                       [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                       [0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0, 1, 1]], dtype=np.uint8)
        self.assertEqual(rel_dist_bound(R1, R2),9)
        return

    def test_example4_18(self):
        R1 = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 1], \
                       [0, 0, 1, 0, 1, 0, 0, 0, 0, 1], \
                       [0, 1, 0, 1, 0, 0, 1, 0, 1, 0], \
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                       [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
        R2 = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 0, 1], \
                       [0, 0, 1, 1, 1, 0, 0, 0, 0, 0], \
                       [0, 1, 0, 1, 0, 0, 0, 0, 1, 0], \
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], \
                       [0, 1, 0, 0, 0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        self.assertEqual(rel_dist_bound(R1, R2),2)
        return

    def test_read_dict_of_lists(self):
        Rdict1 = {
        'a': [False, True, True, True, False, False, False, False, False, False],
        'b': [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        'c': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        'd': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]}
        with self.assertRaises((AssertionError, TypeError)):
            rel_from_dict(Rdict1)
        return

    def test_read_dict_of_dicts(self):
        Rdict2 = {
        'a': {1:False, 2:True, 3:True, 4:True, 5:False, 6:False, 7:False, 8:False, 9:False, 10:False},
        'b': {1:0, 2:0, 3:1, 4:1, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0},
        'c': {1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:0, 8:0, 9:0, 10:0},
        'd': {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1, 10:1}}
        self.assertFalse(np.any(rel_from_dict(Rdict2) - self.R))
        return

    def test_read_sparse_dict(self):
        Rdict3 = {
        'a': {1:False, 2:True, 3:True, 4:True, 5:False, 6:False, 7:False, 8:False, 9:False, 10:False},
        'b': {3:1, 4:1},
        'c': {5:1, 6:1},
        'd': {7:1, 8:1, 9:1, 10:1}}
        self.assertFalse(np.any(rel_from_dict(Rdict3) - self.R))
        return


if __name__ == '__main__':
    unittest.main()
