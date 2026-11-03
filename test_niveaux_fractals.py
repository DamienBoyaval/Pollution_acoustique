import unittest
import tempfile
import numpy as np
import essai_niveaux_fractals as experiment


class FractalLevelTests(unittest.TestCase):
    def test_equal_quantity_and_boundary_data(self):
        with tempfile.TemporaryDirectory() as out:
            report=experiment.run(levels=[0,1],N=16,max_iter=2,output=out)
            rows=report['results']
            self.assertEqual(len(rows),8)
            for row in rows:
                self.assertAlmostEqual(row['material_quantity_discrete'],.4,places=10)
                self.assertLessEqual(row['energy_final_mean'],row['energy_initial_mean'])
                self.assertAlmostEqual(row['energy_final_mean_per_area'],
                                       row['energy_final_mean']/row['area_normalized'])
            self.assertTrue(report['same_boundary_data_between_source_cases'])
            # With the longer fractal wall, fixed total quantity means lower coverage.
            fractions={row['level']:row['boundary_fraction'] for row in rows}
            self.assertGreater(fractions[0],fractions[1])
            for level in [0,1]:
                with np.load(out+f'/level_{level}_ROCK_WOOL_plane.npz') as plane, \
                     np.load(out+f'/level_{level}_ROCK_WOOL_plane_plus_gaussian.npz') as gaussian:
                    for key in ['domain','f_dir','f_neu','f_rob','chi_initial']:
                        np.testing.assert_array_equal(plane[key],gaussian[key])
                    self.assertGreater(np.linalg.norm(gaussian['f']-plane['f']),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
