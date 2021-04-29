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

if __name__ == '__main__':
    unittest.main()
