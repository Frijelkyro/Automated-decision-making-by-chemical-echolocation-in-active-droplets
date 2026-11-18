import numpy as np
import unittest

from storage import build_free_index, pack_conc, unpack_conc


class CompactConcentrationTests(unittest.TestCase):
    def test_pack_unpack_concentration_is_exact_with_zeroed_walls(self):
        maze = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int8)
        concentration = np.array(
            [[1.25, 0.0, -3.5], [0.0, 8.125, 9.75]],
            dtype=np.float32,
        )

        free_index = build_free_index(maze)
        restored = unpack_conc(pack_conc(concentration, free_index), free_index)

        self.assertTrue(
            np.array_equal(restored, np.where(maze != 0, concentration, 0.0))
        )
