"""Verification du choix du pic et du balayage sans reoptimisation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import material_alpha
import preprocessing as pre
import processing as pro
import train_source
import demo_control_polycopie2024 as demo
import test_energie_frequence as experiment


class FixedChiTests(unittest.TestCase):
    def test_fixed_spectrum_matches_direct_solver(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                rows, config = experiment.run(fmax=600., points=3, N=32,
                                               max_iter=2, output=directory)
            peak = int(np.argmax([r['energy_initial'] for r in rows]))
            with np.load(out/'chi_fixe.npz') as fixed, np.load(out/'densites.npz') as densities:
                self.assertEqual(float(fixed['target_frequency_hz']), rows[peak]['frequency_hz'])
                np.testing.assert_array_equal(fixed['chi_fixed'], densities['chi_optimized'][peak])
                np.testing.assert_array_equal(fixed['chi_fixed_binary'], densities['chi_binary'][peak])
                d = fixed['domain']
                bp,ap,ad,bn,_,br = pre._set_coefficients_of_pde(*d.shape)
                _,fd,fn,fr = pre._set_rhs_of_pde(*d.shape)
                omega, alpha, _ = material_alpha.load_material('ROCK_WOOL')
                with (out/'energies_chi_fixe.csv').open() as file:
                    spectrum = list(csv.DictReader(file))
                for index, result in zip(config['frequency_indices'], spectrum):
                    k, A = material_alpha.solver_parameters(omega[index], alpha[index], 1., 340.)
                    source, _ = train_source.gaussian_train_source(d, 1/32, float(result['frequency_hz']))
                    for name, key in [('chi_fixed','energy_fixed'), ('chi_fixed_binary','energy_fixed_binary')]:
                        u = pro.solve_helmholtz(d,1/32,float(k),source,fd,fn,fr,bp,ap,ad,bn,br,A*fixed[name])
                        expected = demo.your_compute_objective_function(d,u,1/32)
                        np.testing.assert_allclose(float(result[key]), expected, rtol=1e-10, atol=0)
                np.testing.assert_allclose(float(spectrum[peak]['energy_fixed']), rows[peak]['energy_optimized'], rtol=1e-10)
            self.assertEqual(json.loads((out/'configuration.json').read_text())['fixed_chi_target_frequency_hz'], rows[peak]['frequency_hz'])
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.replot(directory)
            self.assertGreater((out/'energie_chi_fixe.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
